/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===--------------------------------------------------------------------------*/

DECLARE_TABLE(double, LGAMMA_NEG_ZP, 37*22)
    /* cell 0: m=3 deg=18  x in [-3.5,-3] */
    -0x1.9260dbc9e59afp+1, -0x1.f717cd335a7b3p-53, -0x1.bdbde4f3fc9dbp+2,
    0x1.a26407f7f13dp-1,    0x1.93e4c0d086d61p+0,    -0x1.605a21c33b89dp-2,    0x1.4def510018eafp-1,    -0x1.595d723121678p-2,    0x1.01ab93e453775p-1,    -0x1.790033e647f91p-2,    0x1.e74d57f3ef948p-2,    -0x1.a95d418a4b589p-2,    0x1.fd81610bf7aabp-2,    -0x1.ebb453f62a40bp-2,    0x1.1853f66d96251p-1,    -0x1.195a9f8f0ffc5p-1,    0x1.a13721d212291p-1,    -0x1.71cce4d821c86p-2,    -0x1.25eaf01afddf3p+0,    -0x1.6d0fd409cd2d1p+3,    -0x1.44ee346ef182bp+4,    -0x1.20ae10f9bd1edp+4,
    /* cell 1: m=4 deg=18  x in [-4.5,-4] */
    -0x1.0284e78599581p+2, 0x1.e78c1e9e43cfep-53, -0x1.967c2d3e7adedp+4,
    0x1.629a20aa6854cp+0,    0x1.8a4ef90ed878fp+0,    -0x1.7f6f92c405fadp-4,    0x1.18afa07bc504dp-1,    -0x1.4c0a16f214373p-4,    0x1.667dd0642e87fp-2,    -0x1.49d2321d2f6bbp-4,    0x1.0f8ab01a409f5p-2,    -0x1.4eb21c8206123p-4,    0x1.a3ea25cda8c8bp-3,    -0x1.b35f42d819413p-3,    -0x1.671bca9ef459ep-1,    -0x1.139d0a81cceb5p+2,    -0x1.c624eb7dafbfdp+3,    -0x1.1897697823992p+5,    -0x1.decf6854ee775p+5,    -0x1.1529da38db0b6p+6,    -0x1.8400c0ec0e5d3p+5,    -0x1.028e796dca34ap+4,
    /* cell 2: m=4 deg=18  x in [-4,-3.5] */
    -0x1.fa471547c2fe5p+1, -0x1.70d4561291237p-56, 0x1.65e56533a461p+4,
    0x1.a4b8bccdefc9p+0,    0x1.8a28880247c8fp+0,    0x1.6ce6691bec3dbp-4,    0x1.19d85f4841862p-1,    0x1.79435f76692ffp-4,    0x1.69cadc44ac08ap-2,    0x1.7897e7f8b5ca6p-4,    0x1.13d70e26b518p-2,    0x1.7d026a59c6c3dp-4,    0x1.ba25e52abbcdp-3,    0x1.7167b0f9f4eb1p-3,    -0x1.d4ed64dd316b5p-2,    0x1.b693fedfd77ddp+1,    -0x1.79c481eefc7f1p+3,    0x1.eac621a6198acp+4,    -0x1.b33cbc6a22d77p+5,    0x1.053875e72b18fp+6,    -0x1.7996cafecc9c6p+5,    0x1.03f27b3b1f2cfp+4,
    /* cell 3: m=5 deg=17  x in [-5.5,-5] */
    -0x1.4086a57f0b6d9p+2, -0x1.95262b72ca9cap-55, -0x1.e6b9f97aad953p+6,
    0x1.ae39c1fb52be3p+0,    0x1.8dfbffa39274p+0,    -0x1.7ccbf3125d532p-6,    0x1.14ffddb2a0272p-1,    -0x1.12f39130a33f5p-6,    0x1.5bbf4322b3349p-2,    -0x1.09ab1b637a27ep-6,    0x1.061946ff1012bp-2,    0x1.b4b9462fee24ap-6,    0x1.058666e1ed3e7p-1,    0x1.9eeb0366f6484p+0,    0x1.a5b5417fb7484p+2,    0x1.2ab18e7307553p+4,    0x1.402a374ba4c5ap+5,    0x1.e66d6e81536abp+5,    0x1.f93cbb5147d46p+5,    0x1.40422ce3ce20cp+5,    0x1.8214c32b1569cp+3,    0x0p+0,
    /* cell 4: m=5 deg=18  x in [-5,-4.5] */
    -0x1.3f7577a6eeafdp+2, 0x1.5de5eab7f12cfp-53, 0x1.d9129ea3b874ep+6,
    0x1.bb7ead068cb81p+0,    0x1.8deaecb1b45b6p+0,    0x1.a442703033adap-7,    0x1.1501ec0ce487ep-1,    0x1.19224f9c6724ap-6,    0x1.5bc20699fd302p-2,    0x1.17cd0a937dbf8p-6,    0x1.0045d2cf37d92p-2,    0x1.0edf546e841d8p-5,    0x1.1048409c3bc4p-4,    0x1.b2b04a81a5166p-1,    -0x1.d945d4f0741b2p+1,    0x1.b1f5bf44051f3p+3,    -0x1.1c2d94b29e00dp+5,    0x1.167e68749d942p+6,    -0x1.871b265d7afadp+6,    0x1.78decbd4fb67bp+6,    -0x1.be1cdeb1bec31p+5,    0x1.f5981efb48fb6p+3,
    /* cell 5: m=6 deg=17  x in [-6.5,-6] */
    -0x1.8016b25897c8dp+2, 0x1.27e0f49a4ba72p-54, -0x1.68ef20dc827bdp+9,
    0x1.de520e595d9c1p+0,    0x1.917481c283ba3p+0,    -0x1.c56dc6b5d53fdp-8,    0x1.14ed21935eeeap-1,    -0x1.74e6fd25e0c51p-9,    0x1.5b4bc4a4883ebp-2,    -0x1.21d2b9d6f41c4p-9,    0x1.0875e829265ap-2,    0x1.011f07de8ee8bp-4,    0x1.4618f9d6b45fp-1,    0x1.1501b59756efp+1,    0x1.0705f572248cfp+3,    0x1.658020fc7b339p+4,    0x1.6ecea1481d68ap+5,    0x1.0c3cfb8af8d97p+6,    0x1.0ca73ba7b73d2p+6,    0x1.494a69a251146p+5,    0x1.7f8b20cde78b5p+3,    0x0p+0,
    /* cell 6: m=6 deg=17  x in [-6,-5.5] */
    -0x1.7fe92f591f40dp+2, -0x1.7dd4ed62cbd32p-52, 0x1.670fb0c580989p+9,
    0x1.e08d03b56ed42p+0,    0x1.91725e8bab697p+0,    -0x1.dc9f5b17e8661p-11,    0x1.14ed1832622d7p-1,    0x1.6fa0a7bdf7e54p-9,    0x1.5b4bcc11614c3p-2,    0x1.23b1680612ed8p-9,    0x1.0874f973cc2dep-2,    -0x1.00f293948fafdp-4,    0x1.4604e27ddf6fbp-1,    -0x1.14ecc285b25b4p+1,    0x1.06f68c9297abap+3,    -0x1.656f1f573ff4p+4,    0x1.6ec163d13c34dp+5,    -0x1.0c36018f260e9p+6,    0x1.0ca2d6a7568d9p+6,    -0x1.4947fbcc58a54p+5,    0x1.7f8bcf824c78ep+3,    0x0p+0,
    /* cell 7: m=7 deg=17  x in [-7.5,-7] */
    -0x1.c0033fdedfe1fp+2, 0x1.20bb7d2324678p-52, -0x1.3b203d22c00bbp+12,
    0x1.01ec063d8b325p+1,    0x1.940fe382205b4p+0,    -0x1.bae796f8ac02bp-9,    0x1.14f980ddbc802p-1,    -0x1.b61c1f52635e8p-12,    0x1.5b49c9e2ec704p-2,    0x1.ec0a4dfba9496p-13,    0x1.09183317f98a2p-2,    0x1.1e3490608c673p-4,    0x1.534ef8f5223efp-1,    0x1.22b4739778bbbp+1,    0x1.11104da0a5b6dp+3,    0x1.708a40beea533p+4,    0x1.775facc3ef1fcp+5,    0x1.10bce7957250fp+6,    0x1.0f7aadd650ee9p+6,    0x1.4ad8fc601aebdp+5,    0x1.7f1b2640296e1p+3,    0x0p+0,
    /* cell 8: m=7 deg=17  x in [-7,-6.5] */
    -0x1.bffcbf76b86fp+2, 0x1.853b29347b806p-57, 0x1.3adfbcff37b48p+12,
    0x1.021512691609ap+1,    0x1.940fa897d9cdp+0,    -0x1.4a5b37d566d7ap-9,    0x1.14f97fd70ea3p-1,    0x1.95c9f6328596dp-12,    0x1.5b49c9da129ddp-2,    -0x1.ebe5e66d55d1cp-13,    0x1.09182d3988fb1p-2,    -0x1.1e338d58d9aacp-4,    0x1.534e8263c7434p-1,    -0x1.22b3f951b2b0ep+1,    0x1.110ff45089c53p+3,    -0x1.7089dee77424bp+4,    0x1.775f6115ebap+5,    -0x1.10bcbff4aed63p+6,    0x1.0f7a9505ccabep+6,    -0x1.4ad8eebf9a9fp+5,    0x1.7f1b2a143fe67p+3,    0x0p+0,
    /* cell 9: m=8 deg=17  x in [-8.5,-8] */
    -0x1.000034028b3f9p+3, -0x1.f60cb3cec1cedp-52, -0x1.3b0447f58e709p+15,
    0x1.11fdf6403b36ap+1,    0x1.960fc6d261312p+0,    -0x1.345ab392aadfep-9,    0x1.150179cf181cap-1,    -0x1.eb896f4408117p-15,    0x1.5b4a0e70eabcfp-2,    0x1.36bb73f2a698ep-11,    0x1.09317787a6844p-2,    0x1.228d17cdea166p-4,    0x1.554b30dcb0b3dp-1,    0x1.24c02221d2217p+1,    0x1.128ea07068725p+3,    0x1.722cdc5b2aedcp+4,    0x1.78a34fd24759bp+5,    0x1.11664ae2d1b93p+6,    0x1.0fe4b3ab37844p+6,    0x1.4b132e5256d39p+5,    0x1.7f0accb3fc13dp+3,    0x0p+0,
    /* cell 10: m=8 deg=17  x in [-8,-7.5] */
    -0x1.ffff97f8159cfp+2, -0x1.e54f415a91586p-55, 0x1.3afbb7f13c129p+15,
    0x1.12031e45f8caep+1,    0x1.960fc11516e53p+0,    -0x1.2648c55f3a00dp-9,    0x1.150179baeccb9p-1,    0x1.4cb48f0bd04bcp-15,    0x1.5b4a0e706463cp-2,    -0x1.36cb4562a4d0bp-11,    0x1.0931776e52221p-2,    -0x1.228d13bb9b265p-4,    0x1.554b2ee17721bp-1,    -0x1.24c02017988c5p+1,    0x1.128e9ef353effp+3,    -0x1.722cdaba24fap+4,    0x1.78a34e9004ce8p+5,    -0x1.11664a3a3a909p+6,    0x1.0fe4b341bd12ap+6,    -0x1.4b132e187864ap+5,    0x1.7f0accc43e22ap+3,    0x0p+0,
    /* cell 11: m=9 deg=17  x in [-9.5,-9] */
    -0x1.200005c7768fbp+3, -0x1.b5b610ffb70d4p-54, -0x1.6260901c872f2p+18,
    0x1.20392429c44cep+1,    0x1.97a44f3c7f474p+0,    -0x1.e45687f54bc42p-10,    0x1.1506783a9be38p-1,    -0x1.5b539e292ba8ap-17,    0x1.5b4a28db7fbc3p-2,    0x1.4e9841989433bp-11,    0x1.0934b4aaf68f7p-2,    0x1.231ade4619057p-4,    0x1.558c01c05dfp-1,    0x1.2502dc9b6702cp+1,    0x1.12bf50aedf417p+3,    0x1.7262234a0b135p+4,    0x1.78cc7ad860d92p+5,    0x1.117bd4101552ap+6,    0x1.0ff22cc9bdd16p+6,    0x1.4b1a9281ea3f5p+5,    0x1.7f08b918fb73ep+3,    0x0p+0,
    /* cell 12: m=9 deg=17  x in [-9,-8.5] */
    -0x1.1ffffa3884bdp+3, -0x1.ff90c9d2ae925p-53, 0x1.625f6fe319668p+18,
    0x1.2039b767311aap+1,    0x1.97a44eb9b6ba9p+0,    -0x1.e1361689b686bp-10,    0x1.150678392b47bp-1,    -0x1.e706684a58583p-20,    0x1.5b4a28db79c6p-2,    -0x1.4ea0c6cc973a5p-11,    0x1.0934b4aaa2269p-2,    -0x1.231ade5386f59p-4,    0x1.558c01b9c61fdp-1,    -0x1.2502dc949f28ep+1,    0x1.12bf50a9eb9c5p+3,    -0x1.72622344a00e1p+4,    0x1.78cc7ad431228p+5,    -0x1.117bd40de4b7bp+6,    0x1.0ff22cc85f1c8p+6,    -0x1.4b1a928129dafp+5,    0x1.7f08b919317d1p+3,    0x0p+0,
    /* cell 13: m=10 deg=17  x in [-10.5,-10] */
    -0x1.40000093f2777p+3, -0x1.927b45d95e154p-52, -0x1.baf812d06308p+21,
    0x1.2d0633329314bp+1,    0x1.98ebfd14d62f4p+0,    -0x1.8b8cb179f5f33p-10,    0x1.1509bf16164c9p-1,    -0x1.bb820747c381ap-19,    0x1.5b4a34ad00f39p-2,    0x1.514903be6112dp-11,    0x1.0935121da6a5ap-2,    0x1.232ad3408ee37p-4,    0x1.55934d78e68e1p-1,    0x1.250a5f387ef27p+1,    0x1.12c4cb6b59ac8p+3,    0x1.7268221c20affp+4,    0x1.78d11cbf35e7dp+5,    0x1.117e40646214ap+6,    0x1.0ff3b0d9db9abp+6,    0x1.4b1b67626cc66p+5,    0x1.7f087d4a4ed24p+3,    0x0p+0,
    /* cell 14: m=10 deg=17  x in [-10,-9.5] */
    -0x1.3fffff6c0d7cp+3, 0x1.197cea8c42d7dp-51, 0x1.baf7ed2f9bb55p+21,
    0x1.2d0641f7c0b81p+1,    0x1.98ebfd0a201f1p+0,    -0x1.8b3cb027c608ep-10,    0x1.1509bf15fd8edp-1,    -0x1.463987950a6d8p-18,    0x1.5b4a34ad00a6p-2,    -0x1.514db5a3aacdep-11,    0x1.0935121da5c3ap-2,    -0x1.232ad34cfc651p-4,    0x1.55934d78d4efbp-1,    -0x1.250a5f386d668p+1,    0x1.12c4cb6b4c717p+3,    -0x1.7268221c12366p+4,    0x1.78d11cbf2ab8fp+5,    -0x1.117e40645c3b6p+6,    0x1.0ff3b0d9d7f27p+6,    -0x1.4b1b67626ac58p+5,    0x1.7f087d4a4f643p+3,    0x0p+0,
    /* cell 15: m=11 deg=17  x in [-11.5,-11] */
    -0x1.6000000d7322ap+3, -0x1.8aecb2d37ff52p-51, -0x1.308a8138a9225p+25,
    0x1.38a922a30725bp+1,    0x1.99facc66a13e2p+0,    -0x1.49c1ad9dd4c32p-10,    0x1.150bfc0a129bcp-1,    -0x1.a0d1bdace5ad4p-20,    0x1.5b4a3b0dbc962p-2,    0x1.518f6085268a3p-11,    0x1.09351b963eb3ep-2,    0x1.232c6fe8174p-4,    0x1.55940a247fbcdp-1,    0x1.250b216efcc22p+1,    0x1.12c5591d9a3a9p+3,    0x1.7268bd25b351ep+4,    0x1.78d19489b6bfdp+5,    0x1.117e7f0d816dcp+6,    0x1.0ff3d80cba152p+6,    0x1.4b1b7ce324644p+5,    0x1.7f08773fc510ep+3,    0x0p+0,
    /* cell 16: m=11 deg=17  x in [-11,-10.5] */
    -0x1.5ffffff28cdd4p+3, 0x1.c9924a65aa486p-53, 0x1.308a7ec756dbbp+25,
    0x1.38a923fba91f4p+1,    0x1.99facc65d15abp+0,    -0x1.49ba72b2b5c66p-10,    0x1.150bfc0a110b2p-1,    -0x1.1518bb89da2c6p-18,    0x1.5b4a3b0dbc922p-2,    -0x1.51921aab89111p-11,    0x1.09351b963eb18p-2,    -0x1.232c6fee30f5cp-4,    0x1.55940a247f8cap-1,    -0x1.250b216efccd2p+1,    0x1.12c5591d9a152p+3,    -0x1.7268bd25b327fp+4,    0x1.78d19489b69e2p+5,    -0x1.117e7f0d815b1p+6,    0x1.0ff3d80cba082p+6,    -0x1.4b1b7ce3245aap+5,    0x1.7f08773fc50d3p+3,    0x0p+0,
    /* cell 17: m=12 deg=17  x in [-12.5,-12] */
    -0x1.800000011eed9p+3, 0x1.19d5307e1fb5ep-53, -0x1.c8cfc0286a79cp+28,
    0x1.4353cdeb9ebf3p+1,    0x1.9ade5a9f242f8p+0,    -0x1.172cf5b84c1dap-10,    0x1.150d909521effp-1,    -0x1.7909c2dc04ff3p-21,    0x1.5b4a3ece2a0fp-2,    0x1.5196459768744p-11,    0x1.09351c7894094p-2,    0x1.232c95bcf23cp-4,    0x1.55941b6fffffap-1,    0x1.250b333c89baap+1,    0x1.12c5661ac173bp+3,    0x1.7268cb5bec439p+4,    0x1.78d19f84d11c8p+5,    0x1.117e84cbf1bddp+6,    0x1.0ff3dba4973dap+6,    0x1.4b1b7edbbdc54p+5,    0x1.7f0876b200eccp+3,    0x0p+0,
    /* cell 18: m=12 deg=17  x in [-12,-11.5] */
    -0x1.7ffffffee1127p+3, -0x1.ce1f7906b30f5p-54, 0x1.c8cfbfd795864p+28,
    0x1.4353ce0866d9cp+1,    0x1.9ade5a9f1584bp+0,    -0x1.172c66a4ea2f6p-10,    0x1.150d909521d8p-1,    -0x1.c4964339eeccdp-19,    0x1.5b4a3ece2a0ecp-2,    -0x1.5197edc3e6e2bp-11,    0x1.09351c7894094p-2,    -0x1.232c95c016a9dp-4,    0x1.55941b6fffffcp-1,    -0x1.250b333c89d58p+1,    0x1.12c5661ac173dp+3,    -0x1.7268cb5bec43dp+4,    0x1.78d19f84d11ccp+5,    -0x1.117e84cbf1bdfp+6,    0x1.0ff3dba4973ddp+6,    -0x1.4b1b7edbbdc57p+5,    0x1.7f0876b200edp+3,    0x0p+0,
    /* cell 19: m=13 deg=17  x in [-13.5,-13] */
    -0x1.a000000016124p+3, -0x1.84e03341ee8ddp-51, -0x1.7328cc029a58dp+32,
    0x1.4d2c6b82bf80fp+1,    0x1.9ba03f5af2ebep+0,    -0x1.decd9257c6ce8p-11,    0x1.150eb64a8118cp-1,    -0x1.56f8ce63b0a0ap-23,    0x1.5b4a411f8046ap-2,    0x1.51971c7076bdfp-11,    0x1.09351c8dc8e7ep-2,    0x1.232c98ea42088p-4,    0x1.55941ce38b487p-1,    0x1.250b34bafd873p+1,    0x1.12c56731ca32fp+3,    0x1.7268cc8d3a88fp+4,    0x1.78d1a070b6c1dp+5,    0x1.117e854756832p+6,    0x1.0ff3dbf1c8451p+6,    0x1.4b1b7f0615dddp+5,    0x1.7f0876a61b76fp+3,    0x0p+0,
    /* cell 20: m=13 deg=17  x in [-13,-12.5] */
    -0x1.9fffffffe9edcp+3, 0x1.84f40342d001cp-51, 0x1.7328cbfd65a73p+32,
    0x1.4d2c6b84f753p+1,    0x1.9ba03f5af1f43p+0,    -0x1.decd92d437fb7p-11,    0x1.150eb64a81176p-1,    -0x1.7cd0d8ad1b18cp-19,    0x1.5b4a411f8046ap-2,    -0x1.5198282980ab5p-11,    0x1.09351c8dc8e7ep-2,    -0x1.232c98ebf5cf5p-4,    0x1.55941ce38b487p-1,    -0x1.250b34bafd93ap+1,    0x1.12c56731ca32fp+3,    -0x1.7268cc8d3a88fp+4,    0x1.78d1a070b6c1dp+5,    -0x1.117e854756832p+6,    0x1.0ff3dbf1c8451p+6,    -0x1.4b1b7f0615dddp+5,    0x1.7f0876a61b76fp+3,    0x0p+0,
    /* cell 21: m=14 deg=17  x in [-14.5,-14] */
    -0x1.c000000001939p+3, -0x1.d2a2f4a73af63p-51, -0x1.44c3b2802aca2p+36,
    0x1.5650fdccebacbp+1,    0x1.9c476e602befep+0,    -0x1.9f1d23bd0bad1p-11,    0x1.150f90a744005p-1,    0x1.c84314b825c2dp-23,    0x1.5b4a429bc9c5cp-2,    0x1.519755936abe5p-11,    0x1.09351c90ad5dp-2,    0x1.232c992981b4p-4,    0x1.55941d004c097p-1,    0x1.250b34d895be5p+1,    0x1.12c5674761bdp+3,    0x1.7268cca4da80ap+4,    0x1.78d1a082f7c55p+5,    0x1.117e8550e2e29p+6,    0x1.0ff3dbf7c1666p+6,    0x1.4b1b7f095cafcp+5,    0x1.7f0876a52fce9p+3,    0x0p+0,
    /* cell 22: m=14 deg=17  x in [-14,-13.5] */
    -0x1.bffffffffe6c7p+3, 0x1.d2a30f3dae0fbp-51, 0x1.44c3b27fd535ep+36,
    0x1.5650fdcd144bdp+1,    0x1.9c476e602be08p+0,    -0x1.9f1d3a67bc21cp-11,    0x1.150f90a744004p-1,    -0x1.4af1f4a0c48d8p-19,    0x1.5b4a429bc9c5cp-2,    -0x1.5198042b65e42p-11,    0x1.09351c90ad5dp-2,    -0x1.232c992a7843ep-4,    0x1.55941d004c097p-1,    -0x1.250b34d895c46p+1,    0x1.12c5674761bdp+3,    -0x1.7268cca4da80ap+4,    0x1.78d1a082f7c55p+5,    -0x1.117e8550e2e29p+6,    0x1.0ff3dbf7c1666p+6,    -0x1.4b1b7f095cafcp+5,    0x1.7f0876a52fce9p+3,    0x0p+0,
    /* cell 23: m=15 deg=17  x in [-15.5,-15] */
    -0x1.e0000000001aep+3, -0x1.fcf9ccde8721p-51, -0x1.3077775802bdbp+40,
    0x1.5ed986558729ep+1,    0x1.9cd91113f0be9p+0,    -0x1.6b55108916dbp-11,    0x1.1510365ab21a5p-1,    0x1.fef35b2a6ba9bp-22,    0x1.5b4a439729ccap-2,    0x1.51977310389c6p-11,    0x1.09351c919e493p-2,    0x1.232c992e383e1p-4,    0x1.55941d025cd53p-1,    0x1.250b34dab5afep+1,    0x1.12c56748ee977p+3,    0x1.7268cca68cb88p+4,    0x1.78d1a08447461p+5,    0x1.117e85519261ap+6,    0x1.0ff3dbf82f2fbp+6,    0x1.4b1b7f0998e98p+5,    0x1.7f0876a51ee3fp+3,    0x0p+0,
    /* cell 24: m=15 deg=17  x in [-15,-14.5] */
    -0x1.dfffffffffe52p+3, 0x1.fcf9ccfd8867ep-51, 0x1.30777757fd425p+40,
    0x1.5ed9865589dfcp+1,    0x1.9cd91113f0bdap+0,    -0x1.6b5528cb73dddp-11,    0x1.1510365ab21a5p-1,    -0x1.279939c75e70dp-19,    0x1.5b4a439729ccap-2,    -0x1.5197e833356cbp-11,    0x1.09351c919e493p-2,    -0x1.232c992ec91cdp-4,    0x1.55941d025cd53p-1,    -0x1.250b34dab5b3p+1,    0x1.12c56748ee977p+3,    -0x1.7268cca68cb88p+4,    0x1.78d1a08447461p+5,    -0x1.117e85519261ap+6,    0x1.0ff3dbf82f2fbp+6,    -0x1.4b1b7f0998e98p+5,    0x1.7f0876a51ee3fp+3,    0x0p+0,
    /* cell 25: m=16 deg=17  x in [-16.5,-16] */
    -0x1.000000000000dp+4, -0x1.cfe7ce6768509p-50, -0x1.30777758002cep+44,
    0x1.66d98655886f4p+1,    0x1.9d591113f0be2p+0,    -0x1.40aa65d0c5cebp-11,    0x1.1510b65ab21a5p-1,    0x1.65e0474fa99d4p-21,    0x1.5b4a4441d47a6p-2,    0x1.519785666017fp-11,    0x1.09351c92200dbp-2,    0x1.232c992ea20a9p-4,    0x1.55941d0280654p-1,    0x1.250b34dada18p+1,    0x1.12c5674909271p+3,    0x1.7268cca6a9c8bp+4,    0x1.78d1a0845dbb1p+5,    0x1.117e85519e20ep+6,    0x1.0ff3dbf83688fp+6,    0x1.4b1b7f099cf17p+5,    0x1.7f0876a51dc1bp+3,    0x0p+0,
    /* cell 26: m=16 deg=17  x in [-16,-15.5] */
    -0x1.fffffffffffe5p+3, -0x1.80c18cc43ea26p-53, 0x1.30777757ffd32p+44,
    0x1.66d98655889a5p+1,    0x1.9d591113f0be1p+0,    -0x1.40aa7e2e6f94dp-11,    0x1.1510b65ab21a5p-1,    -0x1.0dffad02c821p-19,    0x1.5b4a4441d47a6p-2,    -0x1.5197d5f713aabp-11,    0x1.09351c92200d2p-2,    -0x1.232c992ef9f28p-4,    0x1.55941d0280567p-1,    -0x1.250b34dada08bp+1,    0x1.12c5674909189p+3,    -0x1.7268cca6a9b64p+4,    0x1.78d1a0845da9fp+5,    -0x1.117e85519e15ap+6,    0x1.0ff3dbf8367efp+6,    -0x1.4b1b7f099ce6cp+5,    0x1.7f0876a51db74p+3,    0x0p+0,
    /* cell 27: m=17 deg=16  x in [-17.5,-17] */
    -0x1.1000000000001p+4, 0x1.ab4e23f3d4bbcp-51, -0x1.437eeecd8002ep+48,
    0x1.6e610ddd1009fp+1,    0x1.9dca736729232p+0,    -0x1.1d1861668af6bp-11,    0x1.1511156252f16p-1,    -0x1.7f9b8388b0975p-18,    0x1.5b204fb219a94p-2,    -0x1.fd53a60368053p-10,    0x1.d6f27821914a7p-3,    -0x1.4e461409ed00fp-3,    -0x1.71c4aa5c11f75p-1,    -0x1.eb83bfef4e6bap+1,    -0x1.739da8367b4ebp+3,    -0x1.a5a03ba7e32bp+4,    -0x1.4e3c9beb19107p+5,    -0x1.691de9c7fe11fp+5,    -0x1.d9bd96b3edaedp+4,    -0x1.27aff2214b34p+3,    0x0p+0,    0x0p+0,
    /* cell 28: m=17 deg=17  x in [-17,-16.5] */
    -0x1.0ffffffffffffp+4, -0x1.ab4e23f3d49f1p-51, 0x1.437eeecd7ffd2p+48,
    0x1.6e610ddd100d7p+1,    0x1.9dca736734f34p+0,    -0x1.1d1831dda8986p-11,    0x1.15111aca98a67p-1,    -0x1.f62f95670401ep-20,    0x1.5b4a44b87465fp-2,    -0x1.5197ca0194b7ap-11,    0x1.09351c926ef75p-2,    -0x1.232c992eee25ep-4,    0x1.55941d02828bep-1,    -0x1.250b34dadc30ep+1,    0x1.12c567490aa9bp+3,    -0x1.7268cca6ab6acp+4,    0x1.78d1a0845ef82p+5,    -0x1.117e85519ec1cp+6,    0x1.0ff3dbf836e7p+6,    -0x1.4b1b7f099d193p+5,    0x1.7f0876a51d97cp+3,    0x0p+0,
    /* cell 29: m=18 deg=16  x in [-18.5,-18] */
    -0x1.2p+4, -0x1.6827863b97d95p-53, -0x1.6beecca730003p+52,
    0x1.757d7fa42c7cfp+1,    0x1.9e2f962b1c7edp+0,    -0x1.fe4215cafe9e2p-12,    0x1.1511654b399ebp-1,    -0x1.78811f5c72a33p-18,    0x1.5b20500648f6dp-2,    -0x1.fd53a20121ffbp-10,    0x1.d6f27821f5056p-3,    -0x1.4e461409e850ep-3,    -0x1.71c4aa5c123dfp-1,    -0x1.eb83bfef4eb57p+1,    -0x1.739da8367b7ddp+3,    -0x1.a5a03ba7e3558p+4,    -0x1.4e3c9beb192afp+5,    -0x1.691de9c7fe271p+5,    -0x1.d9bd96b3edc17p+4,    -0x1.27aff2214b39cp+3,    0x0p+0,    0x0p+0,
    /* cell 30: m=18 deg=16  x in [-18,-17.5] */
    -0x1.2p+4, 0x1.6827863b97d9ap-53, 0x1.6beecca72fffdp+52,
    0x1.757d7fa42c7f5p+1,    0x1.9e2f962b1c7edp+0,    -0x1.fe4126e8e856dp-12,    0x1.1511654b399ebp-1,    0x1.3f59fd2ebacc7p-18,    0x1.5b20500648f6dp-2,    0x1.fd538db399213p-10,    0x1.d6f27821f5056p-3,    0x1.4e461409d6a9ap-3,    -0x1.71c4aa5c123dfp-1,    0x1.eb83bfef4eb4fp+1,    -0x1.739da8367b7ddp+3,    0x1.a5a03ba7e3558p+4,    -0x1.4e3c9beb192afp+5,    0x1.691de9c7fe271p+5,    -0x1.d9bd96b3edc17p+4,    0x1.27aff2214b39cp+3,    0x0p+0,    0x0p+0,
    /* cell 31: m=19 deg=16  x in [-19.5,-19] */
    -0x1.3p+4, -0x1.2f49b46814157p-57, -0x1.b02b930689p+56,
    0x1.7c3a215354e8dp+1,    0x1.9e8a5b4f470b3p+0,    -0x1.cb4cac27dfc08p-12,    0x1.1511a5a9a41bep-1,    -0x1.731577671fac4p-18,    0x1.5b20504325aap-2,    -0x1.fd539f423e252p-10,    0x1.d6f2782235c2fp-3,    -0x1.4e461409e5478p-3,    -0x1.71c4aa5c1232fp-1,    -0x1.eb83bfef4eb36p+1,    -0x1.739da8367b7c5p+3,    -0x1.a5a03ba7e353dp+4,    -0x1.4e3c9beb1929ap+5,    -0x1.691de9c7fe25ap+5,    -0x1.d9bd96b3edbfap+4,    -0x1.27aff2214b38ap+3,    0x0p+0,    0x0p+0,
    /* cell 32: m=19 deg=16  x in [-19,-18.5] */
    -0x1.3p+4, 0x1.2f49b46814157p-57, 0x1.b02b930689p+56,
    0x1.7c3a215354ebp+1,    0x1.9e8a5b4f470b3p+0,    -0x1.cb4bbd45cc5bcp-12,    0x1.1511a5a9a41bep-1,    0x1.44c5a5236031cp-18,    0x1.5b20504325aap-2,    0x1.fd5390727c4c6p-10,    0x1.d6f2782235c2fp-3,    0x1.4e461409d9afp-3,    -0x1.71c4aa5c1232fp-1,    0x1.eb83bfef4eb31p+1,    -0x1.739da8367b7c5p+3,    0x1.a5a03ba7e353dp+4,    -0x1.4e3c9beb1929ap+5,    0x1.691de9c7fe25ap+5,    -0x1.d9bd96b3edbfap+4,    0x1.27aff2214b38ap+3,    0x0p+0,    0x0p+0,
    /* cell 33: m=20 deg=16  x in [-20.5,-20] */
    -0x1.4p+4, -0x1.e542ba4020225p-62, -0x1.0e1b3be415ap+61,
    0x1.82a087b9bb4f3p+1,    0x1.9edc46d465c38p+0,    -0x1.9f9bdca005e7cp-12,    0x1.1511da1769f22p-1,    -0x1.6ee3b97eed9c5p-18,    0x1.5b20506fe2e9p-2,    -0x1.fd539d5763a7bp-10,    0x1.d6f2782260b61p-3,    -0x1.4e461409e35ecp-3,    -0x1.71c4aa5c122d5p-1,    -0x1.eb83bfef4eb33p+1,    -0x1.739da8367b7c4p+3,    -0x1.a5a03ba7e353cp+4,    -0x1.4e3c9beb19299p+5,    -0x1.691de9c7fe259p+5,    -0x1.d9bd96b3edbf8p+4,    -0x1.27aff2214b389p+3,    0x0p+0,    0x0p+0,
    /* cell 34: m=20 deg=16  x in [-20,-19.5] */
    -0x1.4p+4, 0x1.e542ba4020225p-62, 0x1.0e1b3be415ap+61,
    0x1.82a087b9bb516p+1,    0x1.9edc46d465c38p+0,    -0x1.9f9aedbdf2a9fp-12,    0x1.1511da1769f22p-1,    0x1.48f7630b89189p-18,    0x1.5b20506fe2e9p-2,    0x1.fd53925d56c08p-10,    0x1.d6f2782260b61p-3,    0x1.4e461409db979p-3,    -0x1.71c4aa5c122d5p-1,    0x1.eb83bfef4eb3p+1,    -0x1.739da8367b7c4p+3,    0x1.a5a03ba7e353cp+4,    -0x1.4e3c9beb19299p+5,    0x1.691de9c7fe259p+5,    -0x1.d9bd96b3edbf8p+4,    0x1.27aff2214b389p+3,    0x0p+0,    0x0p+0,
    /* cell 35: m=21 deg=16  x in [-21.5,-21] */
    -0x1.5p+4, -0x1.71b8ef6dcf572p-66, -0x1.6283be9b5c62p+65,
    0x1.88b8e93fd3b0bp+1,    0x1.9f26949dd4a3ap+0,    -0x1.79de0077f59c5p-12,    0x1.151205398a1ffp-1,    -0x1.6b9a6b769002fp-18,    0x1.5b205091457fbp-2,    -0x1.fd539bfa8c864p-10,    0x1.d6f278227dc8p-3,    -0x1.4e461409e223cp-3,    -0x1.71c4aa5c1229fp-1,    -0x1.eb83bfef4eb32p+1,    -0x1.739da8367b7c4p+3,    -0x1.a5a03ba7e353cp+4,    -0x1.4e3c9beb19299p+5,    -0x1.691de9c7fe259p+5,    -0x1.d9bd96b3edbf8p+4,    -0x1.27aff2214b389p+3,    0x0p+0,    0x0p+0,
    /* cell 36: m=21 deg=16  x in [-21,-20.5] */
    -0x1.5p+4, 0x1.71b8ef6dcf572p-66, 0x1.6283be9b5c62p+65,
    0x1.88b8e93fd3b2fp+1,    0x1.9f26949dd4a3ap+0,    -0x1.79dd1195e2607p-12,    0x1.151205398a1ffp-1,    0x1.4c40b113e63c6p-18,    0x1.5b205091457fbp-2,    0x1.fd5393ba2de18p-10,    0x1.d6f278227dc8p-3,    0x1.4e461409dcd29p-3,    -0x1.71c4aa5c1229fp-1,    0x1.eb83bfef4eb3p+1,    -0x1.739da8367b7c4p+3,    0x1.a5a03ba7e353cp+4,    -0x1.4e3c9beb19299p+5,    0x1.691de9c7fe259p+5,    -0x1.d9bd96b3edbf8p+4,    0x1.27aff2214b389p+3,    0x0p+0,    0x0p+0
END_TABLE()

DECLARE_TABLE(double, LGAMMA_NEG_S3_ZBP, 10)
    0x1.0642c54e02e46p+1,
    0x1.b8b1c2fc7beccp+0,
    0x1.4fc0d68074a33p-1,
    0x1.e458cc261814ep-1,
    0x1.985b67804f18p-1,
    0x1.f8cda9fb53e91p-1,
    0x1.f24bfb07e3518p-1,
    0x1.f1ab2848d52b6p-1,
    0x1.549fb7e07e002p-1,
    0x1.1d4cc282a2107p-2
END_TABLE()

DECLARE_TABLE(double, LGAMMA_NEG_S3_ZAP, 17)
    -0x1.584ced9c90da6p-1,
    0x1.3b74ee0ed5072p+1,
    -0x1.0a50cbfb7b661p+1,
    0x1.7ee6c6759f38bp+1,
    -0x1.0eb9354f1fe33p+2,
    0x1.a21098888a148p+2,
    -0x1.479d62fcfdfdcp+3,
    0x1.05fa91a150e89p+4,
    -0x1.a0ec1fe86345ap+4,
    0x1.41ca8bdea2523p+5,
    -0x1.ce673bbce409p+5,
    0x1.2696355480711p+6,
    -0x1.3b0aa94ff9484p+6,
    0x1.09d4ae03861f6p+6,
    -0x1.46e3d149a0286p+5,
    0x1.01d4e00c12ff9p+4,
    -0x1.84ec64835bb27p+1
END_TABLE()

DECLARE_TABLE(double, LGAMMA_NEG_S3_ZFB, 24)
    -0x1.ea12da904b18cp+0,
    0x1.3267f3c265a5ap+3,
    -0x1.4185ac30c8e5dp+4,
    0x1.f504accc96cfcp+5,
    -0x1.85884581870abp+7,
    0x1.4373f7d679fbep+9,
    -0x1.12239c23f0da3p+11,
    0x1.dba652bf8e228p+12,
    -0x1.a2d2386f0e1a4p+14,
    0x1.7584e21898cd5p+16,
    -0x1.50688013281b1p+18,
    0x1.310318ea842ecp+20,
    -0x1.19808674325ebp+22,
    0x1.0ad53fd9351c6p+24,
    -0x1.472d78d1d45fcp+25,
    0x1.f98ac822b1251p+27,
    -0x1.343e37ca39065p+32,
    -0x1.8e43d9c03a8dep+35,
    -0x1.0d3473887baacp+31,
    0x1.a1ee662f4f50dp+42,
    0x1.2eeda1bc5f8ep+46,
    0x1.b39ae64c4c131p+48,
    0x1.4a46a22492fdep+50,
    0x1.b582a6721c173p+50
END_TABLE()

DECLARE_TABLE(double, LGAMMA_NEG_S3_ZFA, 13)
    0x1.83fe966af535ep+0,
    0x1.36eebb002f54ap+2,
    0x1.694a6058b1fb7p+0,
    0x1.1718d7ca123b6p+3,
    0x1.7339fd9301702p+2,
    0x1.8d32fb348be14p+4,
    0x1.809fda6553ae7p+4,
    0x1.48d38b4ddb834p+6,
    0x1.95edd60b6555dp+6,
    0x1.179a34bc64b21p+8,
    0x1.395ce913d0312p+9,
    -0x1.0f2a6fceecd39p+8,
    0x1.a4dcdd1f611e6p+12
END_TABLE()

DECLARE_TABLE(double, LGAMMA_NEG_S3_DD, 20)
    -0x1.ccbf9f5ed0f15p-5,
    0x1.1a68793defc15p+0,
    0x1.3141822e16967p+2,
    -0x1.27781d4b3c4aep-6,
    0x1.03a9f1168f38dp+3,
    -0x1.2914a528141bdp-11,
    0x1.55d348c791939p+4,
    -0x1.f620192b25bcfp-16,
    0x1.000a74eeac5bap+6,
    0x1.1a64deeec5b5fp-7,
    0x1.99aa2847c9ac8p+7,
    -0x1.b03bf8d24987ap+1,
    0x1.26cdadac8c93p+9,
    -0x1.4b93f146ba5bdp+10,
    -0x1.33c3b5f156adcp+13,
    -0x1.2a7fe426ddfacp+16,
    -0x1.38bebdd6fe25p+18,
    -0x1.c99f28adbc96cp+19,
    -0x1.8630410830279p+20,
    -0x1.4ba94d40511e5p+20
END_TABLE()

