import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from modules.general_toxicity import train_ld50_model, train_carcinogenicity_model, train_general_tox_model
from modules.tissue_accumulation import train_bbb_model, train_oct2_model, train_vd_model

if __name__ == "__main__":
    train_ld50_model()
    train_carcinogenicity_model()
    train_general_tox_model()
    train_bbb_model()
    train_oct2_model()
    train_vd_model()
