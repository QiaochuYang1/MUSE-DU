import torch

from musedu.echo_mer import ECHOMER


def test_echo_mer_padding_invariance_and_gradients():
    torch.manual_seed(7)
    x = torch.randn(1, 12, 22)
    times = torch.arange(12).float()[None] * .5
    for conditioned,long_valence in ((False,False),(True,False),(True,True)):
        model = ECHOMER(22, hidden=16, event_conditioned=conditioned,long_valence=long_valence)
        short = model(x, times, torch.ones(1, 12, dtype=torch.bool))['mean']
        padded_x = torch.cat((x, torch.randn(1, 6, 22)), dim=1)
        padded_t = torch.arange(18).float()[None] * .5
        valid = torch.arange(18)[None] < 12
        long = model(padded_x, padded_t, valid)['mean']
        torch.testing.assert_close(short, long[:, :12], atol=1e-6, rtol=1e-6)
        assert torch.count_nonzero(long[:, 12:]) == 0
        short.square().mean().backward()
        assert model.persistence.weight.grad is not None
        if conditioned:
            assert model.event_gate[0].weight.grad is not None
        else:
            assert model.constant_gate_logit.grad is not None
        if long_valence:
            assert model.valence_context_first.weight.grad is not None
