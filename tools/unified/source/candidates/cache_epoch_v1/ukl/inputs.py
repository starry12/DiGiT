"""Keep the accepted UKL windows, hot sets, and feature identities."""
from candidates.ukl_training_adapter_v12.inputs import accepted_inputs,arm_inputs,load_window

_original_arm_inputs=arm_inputs

def arm_inputs(value,arm):
    result=_original_arm_inputs(value,arm)
    digit=_original_arm_inputs(value,'digit')
    result['roots']=digit['roots'];result['labels']=digit['labels']
    return result
