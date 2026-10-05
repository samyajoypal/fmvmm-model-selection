"""Explicit JSON transport of nonfinite floats, with no null substitution.

Numeric arrays become nested lists. This preserves their values, not dtype or
array identity. The reserved scalar tag cannot occur as an ordinary input key.
"""
import json
import math
import numpy as np

TAG='__fmvmm_nonfinite_float_v1__'

def encode(value):
    if isinstance(value,np.ndarray):return encode(value.tolist())
    if isinstance(value,np.generic):return encode(value.item())
    if isinstance(value,float) and not math.isfinite(value):
        return {TAG:'nan' if math.isnan(value) else ('+inf' if value>0 else '-inf')}
    if isinstance(value,dict):
        if TAG in value:raise ValueError('Input contains the reserved nonfinite-float tag')
        if any(not isinstance(key,str) for key in value):
            raise TypeError('JSON object keys must be strings')
        return {key:encode(item) for key,item in value.items()}
    if isinstance(value,(tuple,list)):return [encode(item) for item in value]
    return value

def decode(value):
    if isinstance(value,dict):
        if TAG in value:
            if len(value)!=1 or value[TAG] not in ('+inf','-inf','nan'):
                raise ValueError('Malformed nonfinite-float tag')
            return {'+inf':float('inf'),'-inf':float('-inf'),'nan':float('nan')}[value[TAG]]
        return {key:decode(item) for key,item in value.items()}
    if isinstance(value,list):return [decode(item) for item in value]
    return value

def dumps(value,**kwargs):
    if 'allow_nan' in kwargs:raise ValueError('allow_nan is fixed to False')
    return json.dumps(encode(value),allow_nan=False,**kwargs)

def loads(value,**kwargs):
    if 'parse_constant' in kwargs:raise ValueError('parse_constant is reserved')
    def reject(token):raise ValueError('Untagged nonfinite JSON literal: '+token)
    return decode(json.loads(value,parse_constant=reject,**kwargs))
