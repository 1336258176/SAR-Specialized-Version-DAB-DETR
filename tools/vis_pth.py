import torch

state_dict = torch.load(
    r"output\autodl\fine_tune\epoch_20\checkpoint0019.pth", map_location="cpu", weights_only=False
)

print(type(state_dict))

print("=" * 5 + "resume checkpoint" + "=" * 5)
for name in state_dict:
    print(name)
    # if name == "lr_scheduler":
    #     print(state_dict[name])


pretrain_state_dict = torch.load(r"pth\R50\checkpoint.pth", map_location="cpu", weights_only=False)

print("=" * 5 + "pretrain checkpoint" + "=" * 5)
for name in pretrain_state_dict:
    print(name)
