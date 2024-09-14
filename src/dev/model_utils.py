import os
import torch

def save_model(GP_model, model_file_name='change_this_model_name.pth'):

    file_path = os.path.join('../models', model_file_name)

    torch.save(GP_model.state_dict(), file_path)

    print(f"Model {model_file_name} successfully saved.")
    

def load_model(GP_model, model_file_name):
    #TODO: Will this required adding extra args if there's more specifications to the precise GP model to build?

    state_dict = torch.loard(f'../models/{model_file_name}')

    model = GP_model()

    model.load_state_dict(state_dict)

    return model
