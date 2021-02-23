from conv_block import norm_act, conv_norm, block_conv_attn


def down_stack(x, fltrs=[8,16,16], strides=2, self_attn=[False,False,False],
               fscales=[4,4,4], down_attn=False, squeeze_attn=True, dp_rate=0.2,
               fcnv_scut=False, activation=tf.nn.relu):
    if isinstance(self_attn, bool):
            self_attn = [self_attn]*3
    pos_emb = CONFIG.pos_emb
    frac_dk = CONFIG.frac_dk
    frac_dv = CONFIG.frac_dv

    x = block_conv_attn(x, fltrs[0], strides=strides, activation=activation, frac_dk=frac_dk,
                        conv_shortcut=fcnv_scut, pos_emb=pos_emb, squeeze_attn=squeeze_attn,
                        self_attn=self_attn[0], fscale=fscales[0], down_attn=down_attn,
                        dp_rate=dp_rate, make_model=False, frac_dv=frac_dv)

    x = block_conv_attn(x, fltrs[1], strides=1, activation=activation, frac_dk=frac_dk,
                        conv_shortcut=True, pos_emb=pos_emb, squeeze_attn=squeeze_attn,
                        self_attn=self_attn[1], fscale=fscales[1], down_attn=down_attn,
                        dp_rate=dp_rate, make_model=False, frac_dv=frac_dv)

    x = block_conv_attn(x, fltrs[2], strides=1, activation=activation, frac_dk=frac_dk,
                        conv_shortcut=True, pos_emb=pos_emb, squeeze_attn=squeeze_attn,
                        self_attn=self_attn[2], fscale=fscales[2], down_attn=down_attn,
                        dp_rate=dp_rate, make_model=False, frac_dv=frac_dv)
    return x