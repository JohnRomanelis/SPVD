def print_stats(var, name='_'):
    print(f'{name} | mean: {var.mean().item():.3f} | var: {var.var().item():.3f} | shape: {var.shape}') 
    
def model_num_params(model):
    return sum(
        param.numel() for param in model.parameters()
    )
