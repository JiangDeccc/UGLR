"""Training-only last-text-token codebook initialization for Amazon edu3/age5.

No user examples or labels are encoded. Pairwise penalties act only during
initialization; the model and training objective remain the sqrt-Gini baseline.
"""
import math
import torch
import torch.nn.functional as F

AMAZON_LABELS = {
    'gender': ['male', 'female', 'another gender identity'],
    'education': ['high school diploma or GED', "bachelor's degree",
                  'graduate or professional degree'],
    'age': ['18 to 24 years old', '25 to 34 years old', '35 to 44 years old',
            '45 to 54 years old', '55 years old or older'],
    'income': ['less than $25,000 per year', '$25,000 to $49,999 per year',
               '$50,000 to $74,999 per year', '$75,000 to $99,999 per year',
               '$100,000 to $149,999 per year', '$150,000 or more per year'],
}
ML1M_LABELS = {'gender': ['male', 'female'], 'occupation': ['other/not specified', 'academic/educator', 'artist', 'clerical/admin', 'college/grad student', 'customer service', 'doctor/health care', 'executive/managerial', 'farmer', 'homemaker', 'K-12 student', 'lawyer', 'programmer', 'retired', 'sales/marketing', 'scientist', 'self-employed', 'technician/engineer', 'tradesman/craftsman', 'unemployed', 'writer'], 'age': ['under 18 years old', '18 to 24 years old', '25 to 34 years old', '35 to 44 years old', '45 to 49 years old', '50 to 55 years old', '56 years old or older']}

INIT_VERSION = 'last_text_global_center_norm_match_v1'


def unit_rows(x):
    if not torch.isfinite(x).all() or (x.norm(dim=-1) <= 1e-8).any():
        raise ValueError('Nonfinite or degenerate semantic embeddings; inspect descriptions/centering')
    return F.normalize(x, dim=-1)


def masks(groups):
    g = torch.tensor(groups)
    same = g[:, None] == g[None, :]
    upper = torch.ones_like(same).triu(1)
    return same & upper, ~same & upper


def similarity_penalty(e, groups, within=.5, cross=.3):
    sim = e @ e.T
    a, b = masks(groups)
    return sum((sim[m] - cap).clamp_min(0).square().mean()
               if m.any() else sim.sum() * 0 for m, cap in [(a, within), (b, cross)])


def spread_directions(v, groups, weight=.1, steps=200, lr=.05, within=.5, cross=.3):
    """CPU-only deterministic optimization, preserving semantic directions."""
    v = v.detach().float().cpu()
    with torch.enable_grad():
        x = torch.nn.Parameter(v.clone())
        optimizer = torch.optim.Adam([x], lr=lr)
        best, best_loss = v.clone(), float('inf')
        for _ in range(steps + 1):
            e = F.normalize(x, dim=-1)
            anchor = (e - v).square().sum(-1).mean()
            loss = anchor + weight * similarity_penalty(e, groups, within, cross)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite semantic initialization objective')
            if loss.item() < best_loss:
                best, best_loss = e.detach().clone(), loss.item()
            if _ == steps:
                break
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return unit_rows(best)


def pair_stats(e, groups):
    sim = e @ e.T
    out = {}
    for name, mask in zip(('within', 'cross'), masks(groups)):
        if mask.any():
            out[name + '_mean_cosine'] = sim[mask].mean().item()
            out[name + '_max_cosine'] = sim[mask].max().item()
    return out


def validate_options(args):
    for key in ('semantic_center_strength', 'semantic_similarity_weight',
                'semantic_similarity_lr', 'semantic_within_cap', 'semantic_cross_cap'):
        if not math.isfinite(getattr(args, key)):
            raise ValueError(f'{key} must be finite')
    if not 0 <= args.semantic_center_strength <= 1:
        raise ValueError('semantic_center_strength must be in [0,1]')
    if args.semantic_similarity_weight < 0 or args.semantic_similarity_lr <= 0 or args.semantic_similarity_steps < 0:
        raise ValueError('Invalid semantic similarity optimization settings')
    if not -1 <= args.semantic_within_cap <= 1 or not -1 <= args.semantic_cross_cap <= 1:
        raise ValueError('Cosine caps must be in [-1,1]')


def initialize_codebooks(model, tokenizer, device, args, spread=False, dataset="amazon"):
    labels = {"amazon": AMAZON_LABELS, "ml1m": ML1M_LABELS}[dataset]
    PROMPTS = {a: [[f'{a.capitalize()}: {value}.',
                    f'The user has the following {a}: {value}.'] for value in values]
               for a, values in labels.items()}
    validate_options(args)
    if getattr(model, '_semantic_initialized', False):
        raise RuntimeError('Semantic initialization must run only once before training')
    for attr in model.attr_order:
        if model.num_classes_map[attr] != len(PROMPTS[attr]):
            raise ValueError(f'Semantic label map mismatch for {attr}: expected {len(PROMPTS[attr])}')
    states = [(module, module.training) for module in model.modules()]
    vectors, groups, norms = [], [], {}
    # eval + fork_rng prevents initialization from changing training dropout/RNG state.
    init_device = torch.device(device)
    cuda_devices = ([init_device.index if init_device.index is not None else torch.cuda.current_device()]
                    if init_device.type == 'cuda' else [])
    try:
        model.eval()
        with torch.random.fork_rng(devices=cuda_devices), torch.no_grad():
            for group, attr in enumerate(model.attr_order):
                norms[attr] = getattr(model, f'{attr}_head').codebook.weight.float().norm(dim=-1).mean().item()
                for descriptions in PROMPTS[attr]:
                    encoded = []
                    for text in descriptions:
                        batch = tokenizer(text, return_tensors='pt', add_special_tokens=False,
                                          truncation=False, padding=False)
                        ids = batch['input_ids'].to(device)
                        mask = batch['attention_mask'].to(device)
                        positions = torch.arange(mask.shape[1], device=device).expand_as(mask)
                        last = positions.masked_fill(~mask.bool(), -1).max(-1).values
                        if (last < 0).any():
                            raise ValueError('Empty semantic description')
                        hidden = model.encode_backbone(ids, mask)
                        encoded.append(hidden[torch.arange(len(ids), device=device), last][0].float().cpu())
                    vectors.append(torch.stack(encoded).mean(0))
                    groups.append(group)
    finally:
        for module, training in states:
            module.training = training
    raw = torch.stack(vectors)
    v = unit_rows(raw - args.semantic_center_strength * raw.mean(0, keepdim=True))
    e = spread_directions(v, groups, args.semantic_similarity_weight,
                          args.semantic_similarity_steps, args.semantic_similarity_lr,
                          args.semantic_within_cap, args.semantic_cross_cap) if spread else v
    offset = 0
    with torch.no_grad():
        for attr in model.attr_order:
            weight = getattr(model, f'{attr}_head').codebook.weight
            n = len(weight)
            weight.copy_((e[offset:offset+n] * norms[attr]).to(weight))
            offset += n
    model._semantic_initialized = True
    return dict(semantic_init_version=INIT_VERSION, semantic_pooling='last_valid_text_token',
                semantic_add_special_tokens=False, semantic_prompt_ensemble='mean_of_two_last_hidden_states',
                semantic_center_scope='all_attribute_class_prototypes',
                semantic_norm_matching='per_attribute_preinit_mean_row_norm',
                semantic_similarity_penalty=bool(spread), semantic_prompts=PROMPTS,
                semantic_target_norms=norms, semantic_before=pair_stats(unit_rows(raw), groups),
                semantic_centered=pair_stats(v, groups), semantic_after=pair_stats(e, groups),
                **{k: getattr(args, k) for k in ('semantic_center_strength', 'semantic_similarity_weight',
                   'semantic_similarity_steps', 'semantic_similarity_lr', 'semantic_within_cap', 'semantic_cross_cap')})
