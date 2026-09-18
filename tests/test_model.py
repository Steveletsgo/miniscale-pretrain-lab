import torch

from miniscale.model import DecoderConfig, DecoderLM


def tiny_config() -> DecoderConfig:
    return DecoderConfig(
        vocab_size=128,
        num_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        num_kv_heads=2,
        intermediate_size=128,
        max_sequence_length=16,
    )


def test_forward_shape_and_finite_loss() -> None:
    model = DecoderLM(tiny_config())
    tokens = torch.randint(0, 128, (2, 16))
    output = model(tokens, tokens)
    assert output["logits"].shape == (2, 16, 128)
    assert torch.isfinite(output["loss"])


def test_tied_embeddings() -> None:
    model = DecoderLM(tiny_config())
    assert model.lm_head.weight.data_ptr() == model.token_embedding.weight.data_ptr()


def test_parameter_count_is_positive() -> None:
    assert DecoderLM(tiny_config()).parameter_count() > 0

