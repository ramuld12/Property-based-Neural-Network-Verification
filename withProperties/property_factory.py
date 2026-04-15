import property_driven_ml.logics as pml_logics
from properties_godel import build_properties as build_godel_properties
from properties_dl2 import build_properties as build_dl2_properties

def build_properties(
    logic: pml_logics,
    device,
    scaler,
    feature_names,
    label_encoder,
):
    logic_name = logic.name
    print(f"Building properties for logic: {logic_name}")
    if logic_name == pml_logics.GoedelFuzzyLogic().name:
        return build_godel_properties(device, scaler, feature_names, label_encoder)

    if logic_name == pml_logics.DL2().name:
        return build_dl2_properties(device, scaler, feature_names, label_encoder)

    raise ValueError(f"Unknown property logic: {logic.name}")