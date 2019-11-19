#!/usr/bin/env bash

python main.py --code "CAS_Res3D_34_G3_CVL_S16_B64_S224_TSN_CROP" --option "./options/casargs01.json"

sleep 10s

python main.py --code "CAS_Res3D_34_G3_CVL_S16_B64_S224_TSN_CROP" --option "./options/cas_args021.json"

#sleep 10s
#
#python main.py --code "PKU_Res3D_34_G3_CVM_S16_B64_S224" --option "./options/pku_cv_m.json"
#
#sleep 10s
#
#python main.py --code "PKU_Res3D_34_G3_CS_S16_B64_S224" --option "./options/pku_cs.json"