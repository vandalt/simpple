import inspect
import numpy as np
from scipy.stats._distn_infrastructure import rv_continuous_frozen


def scipy_dist_to_dict(dist):
    dist_dict = dist.__dict__
    comp_dict = {}
    for k in dist_dict:
        if k == "dist":
            comp_dict["scipy_dist_type"] = type(dist_dict[k])
            continue
        elif isinstance(dist_dict[k], dict):
            for kwd in dist_dict[k]:
                comp_dict[f"scipy_dist_{k}_{kwd}"] = dist_dict[k][kwd]
        else:
            comp_dict[f"scipy_dist_{k}"] = dist_dict[k]
    return comp_dict


def make_hashable(obj):
    # vibe-coded
    if isinstance(obj, dict):
        return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
    elif isinstance(obj, (list, tuple)):
        return tuple(make_hashable(x) for x in obj)
    elif isinstance(obj, np.ndarray):
        return obj.tobytes()
    elif isinstance(obj, rv_continuous_frozen):
        return make_hashable(scipy_dist_to_dict(obj))
    else:
        return obj


def find_args(obj, argtype: str = "args"):
    def check_pval_type(pval, argtype: str):
        if argtype == "args":
            return pval.default is pval.empty
        elif argtype == "kwargs":
            return pval.default is not pval.empty
        else:
            raise ValueError("argtype must be one of 'args' or 'kwargs'")

    sig = inspect.signature(obj.__class__.__init__)
    ignored_args = ["self", "args", "kwargs", "parameters"]
    required_args = [
        pname
        for pname, pval in sig.parameters.items()
        if pname not in ignored_args and check_pval_type(pval, argtype)
    ]
    return required_args
