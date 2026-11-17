/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===--------------------------------------------------------------------------*/

DECLARE_TABLE(float, LGAMMAF_NEG_ZP, 19*11)
    /* cell 0: m=3 deg=7  x in [-3.5,-3] */
    -0x1.9260dcp+1f, 0x1.b0d328p-26f, -0x1.bdbde6p+2f,
    0x1.a26402p-1f,    0x1.93e4e8p+0f,    -0x1.6040fep-2f,    0x1.4dc2acp-1f,    -0x1.6376e6p-2f,    0x1.fcd6eap-2f,    -0x1.827d9ep-4f,    0x1.5c0b8p+0f,
    /* cell 1: m=4 deg=7  x in [-4.5,-4] */
    -0x1.0284e8p+2f, 0x1.e99aap-24f, -0x1.967c2ep+4f,
    0x1.629a2p+0f,    0x1.8a4e76p+0f,    -0x1.7f6f44p-4f,    0x1.1af53p-1f,    -0x1.370ba4p-6f,    0x1.64a3c6p-1f,    0x1.a41292p-1f,    0x1.3ad11cp+0f,
    /* cell 2: m=4 deg=7  x in [-4,-3.5] */
    -0x1.fa4716p+1f, 0x1.707a04p-24f, 0x1.65e566p+4f,
    0x1.a4b8bcp+0f,    0x1.8a280ap+0f,    0x1.6d266cp-4f,    0x1.1b7ed8p-1f,    0x1.3c5c36p-5f,    0x1.58abf2p-1f,    -0x1.8e34b2p-1f,    0x1.3c5a88p+0f,
    /* cell 3: m=5 deg=6  x in [-5.5,-5] */
    -0x1.4086a6p+2f, 0x1.01e924p-23f, -0x1.e6b9fap+6f,
    0x1.ae39c6p+0f,    0x1.8df5dp+0f,    -0x1.bd6f56p-6f,    0x1.f5b73ap-2f,    -0x1.3ad80ep-2f,    -0x1.cb1656p-2f,    -0x1.e1ac0ap-1f,    0x0p+0f,
    /* cell 4: m=5 deg=7  x in [-5,-4.5] */
    -0x1.3f7578p+2f, 0x1.64454p-24f, 0x1.d9129ep+6f,
    0x1.bb7eaep+0f,    0x1.8debdp+0f,    0x1.88335cp-7f,    0x1.1cf236p-1f,    -0x1.bc4d26p-4f,    0x1.b67be4p-1f,    -0x1.0f4ce8p+0f,    0x1.323d28p+0f,
    /* cell 5: m=6 deg=6  x in [-6.5,-6] */
    -0x1.8016b2p+2f, -0x1.625f24p-24f, -0x1.68ef2p+9f,
    0x1.de5204p+0f,    0x1.916a5cp+0f,    -0x1.881444p-7f,    0x1.ed661ap-2f,    -0x1.46f80ap-2f,    -0x1.ee03p-2f,    -0x1.de77fcp-1f,    0x0p+0f,
    /* cell 6: m=6 deg=6  x in [-6,-5.5] */
    -0x1.7fe93p+2f, 0x1.4dc17ep-23f, 0x1.670fbp+9f,
    0x1.e08d0ep+0f,    0x1.91683ap+0f,    0x1.0f11e4p-8f,    0x1.ed6856p-2f,    0x1.46e634p-2f,    -0x1.edf99cp-2f,    0x1.de78c4p-1f,    0x0p+0f,
    /* cell 7: m=7 deg=6  x in [-7.5,-7] */
    -0x1.c0034p+2f, 0x1.0900fp-25f, -0x1.3b203ep+12f,
    0x1.01ecp+1f,    0x1.9404f2p+0f,    -0x1.1ae54p-7f,    0x1.ec01aap-2f,    -0x1.492dp-2f,    -0x1.f4074ap-2f,    -0x1.ddea7ep-1f,    0x0p+0f,
    /* cell 8: m=7 deg=6  x in [-7,-6.5] */
    -0x1.bffccp+2f, 0x1.128f22p-23f, 0x1.3adfbcp+12f,
    0x1.02151ap+1f,    0x1.9404b8p+0f,    0x1.66513cp-9f,    0x1.ec01b4p-2f,    0x1.4924c2p-2f,    -0x1.f40716p-2f,    0x1.ddea7ap-1f,    0x0p+0f,
    /* cell 9: m=8 deg=6  x in [-8.5,-8] */
    -0x1.000034p+3f, -0x1.459fcap-28f, -0x1.3b0448p+15f,
    0x1.11fdfp+1f,    0x1.9604b8p+0f,    -0x1.f4886cp-8f,    0x1.ebd96ep-2f,    -0x1.497f22p-2f,    -0x1.f4e81p-2f,    -0x1.ddd5d2p-1f,    0x0p+0f,
    /* cell 10: m=8 deg=6  x in [-8,-7.5] */
    -0x1.ffff98p+2f, 0x1.fa98c4p-28f, 0x1.3afbb8p+15f,
    0x1.120326p+1f,    0x1.9604b2p+0f,    0x1.8e6d5ap-9f,    0x1.ebd96ep-2f,    0x1.497a2ap-2f,    -0x1.f4e80ep-2f,    0x1.ddd5cep-1f,    0x0p+0f,
    /* cell 11: m=9 deg=6  x in [-9.5,-9] */
    -0x1.200006p+3f, 0x1.c44b82p-24f, -0x1.62609p+18f,
    0x1.20391cp+1f,    0x1.97993cp+0f,    -0x1.d3b26ep-8f,    0x1.ebdc46p-2f,    -0x1.4988e4p-2f,    -0x1.f5048cp-2f,    -0x1.ddd332p-1f,    0x0p+0f,
    /* cell 12: m=9 deg=6  x in [-9,-8.5] */
    -0x1.1ffffap+3f, -0x1.c425e8p-24f, 0x1.625f7p+18f,
    0x1.2039bep+1f,    0x1.97993cp+0f,    0x1.c49e8cp-9f,    0x1.ebdc46p-2f,    0x1.4985b4p-2f,    -0x1.f5048cp-2f,    0x1.ddd33p-1f,    0x0p+0f,
    /* cell 13: m=10 deg=6  x in [-10.5,-10] */
    -0x1.4p+3f, -0x1.27e4eep-22f, -0x1.baf814p+21f,
    0x1.2d062cp+1f,    0x1.98e0eap+0f,    -0x1.bd875ep-8f,    0x1.ebe206p-2f,    -0x1.498992p-2f,    -0x1.f507b8p-2f,    -0x1.ddd2e6p-1f,    0x0p+0f,
    /* cell 14: m=10 deg=6  x in [-10,-9.5] */
    -0x1.4p+3f, 0x1.27e508p-22f, 0x1.baf7eep+21f,
    0x1.2d064ap+1f,    0x1.98e0eap+0f,    0x1.efaa0ap-9f,    0x1.ebe206p-2f,    0x1.49876ep-2f,    -0x1.f507b8p-2f,    0x1.ddd2e6p-1f,    0x0p+0f,
    /* cell 15: m=11 deg=6  x in [-11.5,-11] */
    -0x1.6p+3f, -0x1.ae6454p-26f, -0x1.308a82p+25f,
    0x1.38a91cp+1f,    0x1.99efb8p+0f,    -0x1.ad155cp-8f,    0x1.ebe66ap-2f,    -0x1.49895ep-2f,    -0x1.f50804p-2f,    -0x1.ddd2dep-1f,    0x0p+0f,
    /* cell 16: m=11 deg=6  x in [-11,-10.5] */
    -0x1.6p+3f, 0x1.ae6458p-26f, 0x1.308a7ep+25f,
    0x1.38a92cp+1f,    0x1.99efb8p+0f,    0x1.083654p-8f,    0x1.ebe66ap-2f,    0x1.4987ep-2f,    -0x1.f50804p-2f,    0x1.ddd2dep-1f,    0x0p+0f,
    /* cell 17: m=12 deg=6  x in [-12.5,-12] */
    -0x1.8p+3f, -0x1.1eed8ep-29f, -0x1.c8cfc2p+28f,
    0x1.4353c6p+1f,    0x1.9ad346p+0f,    -0x1.a0704p-8f,    0x1.ebe992p-2f,    -0x1.49892ap-2f,    -0x1.f50808p-2f,    -0x1.ddd2dep-1f,    0x0p+0f,
    /* cell 18: m=12 deg=6  x in [-12,-11.5] */
    -0x1.8p+3f, 0x1.1eed9p-29f, 0x1.c8cfbep+28f,
    0x1.4353d6p+1f,    0x1.9ad346p+0f,    0x1.14d9e8p-8f,    0x1.ebe992p-2f,    0x1.498818p-2f,    -0x1.f50808p-2f,    0x1.ddd2dep-1f,    0x0p+0f
END_TABLE()

DECLARE_TABLE(float, LGAMMAF_NEG_S3_ZBP, 4)
    0x1.06370ap+1f,
    0x1.b6866p+0f,
    0x1.27f524p-1f,
    0x1.322f5ep-1f
END_TABLE()

DECLARE_TABLE(float, LGAMMAF_NEG_S3_ZAP, 10)
    -0x1.584cbep-1f,
    0x1.3b72bp+1f,
    -0x1.0a206ep+1f,
    0x1.7c881p+1f,
    -0x1.05127p+2f,
    0x1.6bfb6ep+2f,
    -0x1.b70e7ep+2f,
    0x1.9d7daep+2f,
    -0x1.fdc17cp+1f,
    0x1.2eb17p+0f
END_TABLE()

DECLARE_TABLE(float, LGAMMAF_NEG_S3_ZFB, 11)
    -0x1.ea12dcp+0f,
    0x1.3267fcp+3f,
    -0x1.41847ap+4f,
    0x1.f4f0b2p+5f,
    -0x1.868cacp+7f,
    0x1.45afc8p+9f,
    -0x1.d66252p+10f,
    0x1.1892eap+13f,
    -0x1.456a28p+16f,
    -0x1.46ff6cp+19f,
    -0x1.a6842cp+21f
END_TABLE()

DECLARE_TABLE(float, LGAMMAF_NEG_S3_ZFA, 6)
    0x1.83fe96p+0f,
    0x1.36eeap+2f,
    0x1.693c76p+0f,
    0x1.18428ep+3f,
    0x1.3122fcp+2f,
    0x1.1911bcp+5f
END_TABLE()

DECLARE_TABLE(float, LGAMMAF_NEG_S3_DD, 10)
    -0x1.ccbfap-5f,
    0x1.1a6878p+0f,
    0x1.31418ap+2f,
    -0x1.2458f4p-6f,
    0x1.03b236p+3f,
    -0x1.f5712p-4f,
    0x1.26531cp+4f,
    -0x1.df28dp+4f,
    -0x1.57834ap+6f,
    -0x1.56a788p+8f
END_TABLE()

