# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .rftr import build_rftr

def build_model(args):
    return build_rftr(args)
    
