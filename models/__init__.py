# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from .rftr import build_rftr

def build_model(args):
    #if args.backbone == 'rftr':
    return build_rftr(args)
    #else:
    #    return build(args)
