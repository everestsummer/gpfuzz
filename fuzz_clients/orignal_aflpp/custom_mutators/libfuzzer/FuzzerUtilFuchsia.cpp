//===- FuzzerUtilFuchsia.cpp - Misc utils for Fuchsia. --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Misc utils implementation using Fuchsia/Zircon APIs.
//===----------------------------------------------------------------------===//
#include "FuzzerPlatform.h"

#if LIBFUZZER_FUCHSIA

  #include "FuzzerInternal.h"
  #include "FuzzerUtil.h"
  #include <cassert>
  #include <cerrno>
  #include <cinttypes>
  #include <cstdint>
  #include <fcntl.h>
  #include <lib/fdio/fdio.h>
  #include <lib/fdio/spawn.h>
  #include <string>
  #include <sys/select.h>
  #include <thread>
  #include <unistd.h>
  #include <zircon/errors.h>
  #include <zircon/process.h>
  #include <zircon/sanitizer.h>
  #include <zircon/status.h>
  #include <zircon/syscalls.h>
  #include <zircon/syscalls/debug.h>
  #include <zircon/syscalls/exception.h>
  #include <zircon/syscalls/object.h>
  #include <zircon/types.h>

  #include <vector>

namespace fuzzer {

// Given that Fuchsia doesn't have the POSIX signals that libFuzzer was written
// around, the general approach is to spin up dedicated threads to watch for
// each requested condition (alarm, interrupt, crash).  Of these, the crash
// handler is the most involved, as it requires resuming the crashed thread in
// order to invoke the sanitizers to get the needed state.

// Forward declaration of assembly trampoline needed to resume crashed threads.
// This appears to have external linkage to  C++, which is why it's not in the
// anonymous namespace.  The assembly definition inside MakeTrampoline()
// actually defines the symbol with internal linkage only.
void CrashTrampolineAsm() __asm__("CrashTrampolineAsm");

namespace {

// Helper function to handle Zircon syscall failures.
void ExitOnErr(zx_status_t Status, const char *Syscall) {

  if (Status != ZX_OK) {

    Printf("libFuzzer: %s failed: %s\n", Syscall,
           _zx_status_get_string(Status));
    exit(1);

  }

}

void AlarmHandler(int Seconds) {

  while (true) {

    SleepSeconds(Seconds);
    Fuzzer::StaticAlarmCallback();

  }

}

void InterruptHandler() {

  fd_set readfds;
  // Ctrl-C sends ETX in Zircon.
  do {

    FD_ZERO(&readfds);
    FD_SET(STDIN_FILENO, &readfds);
    select(STDIN_FILENO + 1, &readfds, nullptr, nullptr, nullptr);

  } while (!FD_ISSET(STDIN_FILENO, &readfds) || getchar() != 0x03);

  Fuzzer::StaticInterruptCallback();

}

  // CFAOffset is used to reference the stack pointer before entering the
  // trampoline (Stack Pointer + CFAOffset = prev Stack Pointer). Before jumping
  // to the trampoline we copy all the registers onto the stack. We need to make
  // sure that the new stack has enough space to store all the registers.
  //
  // The trampoline holds CFI information regarding the registers stored in the
  // stack, which is then used by the unwinder to restore them.
  #if defined(__x86_64__)
// In x86_64 the crashing function might also be using the red zone (128 bytes
// on top of their rsp).
constexpr size_t CFAOffset = 128 + sizeof(zx_thread_state_general_regs_t);
  #elif defined(__aarch64__)
// In aarch64 we need to always have the stack pointer aligned to 16 bytes, so
// we make sure that we are keeping that same alignment.
constexpr size_t CFAOffset =
    (sizeof(zx_thread_state_general_regs_t) + 15) & -(uintptr_t)16;
  #endif

  // For the crash handler, we need to call Fuzzer::StaticCrashSignalCallback
  // without POSIX signal handlers.  To achieve this, we use an assembly
  // function to add the necessary CFI unwinding information and a C function to
  // bridge from that back into C++.

  // FIXME: This works as a short-term solution, but this code really shouldn't
  // be architecture dependent. A better long term solution is to implement
  // remote unwinding and expose the necessary APIs through sanitizer_common
  // and/or ASAN to/�G�z̒X��������IU��ݻJ-L��$��{Iw��\�(ڌz�k^���lw�Ґ�ǟ
�Դ{I�}�̽w
��| ���˅X�2�=�W�e���H�d��ӄ����>�g��HXH]B.Q>�٪�_�"5-,u1�8dr���1,`S\
)�+�k��]�/9xC�'�j����&DDA�ʠ`Y�<�x�1�FWv�����gR�j��ٯ<��~��W�g��C�i��y�	���{��L�����Qg���H�R
K��j(4�{���$�2�<�����:�|���9�v4�Y\f�u(OJ�]bn�\O��u4�%T�o���M��{D�7�Y�Y��������l����:[C\ :�2�(F+@A�
c-W�ώ@�au��C5&�5T�R&hW�+�m�e��h��Υ}Z�.�.�'�2�B	f��{������9y�ϥ4,Z!4��!����K�Z-<jm����SW!�ϭ+ $�[�b1�Fv�׏��x���(��i��7�Y}(��W�ρB�o���f��Z���:�U�z���4��O\��Ch��%�<�;�ԶW�����ד�����ؤ39Iw�n���f`̕-�+O�B�nI�T���!zB�~�ە��ۿ�`q��-tڏm/}e���7�j%��I�V�����z��Ǡs��)�2��KWZ?�`��.\m8�4�&*��ܕ�Wl틏��1dE�9!�̼p2�ju�ݩ.��¼Unԅ&����a���=�lֈ�������ЗYn5�_�!�$�u3��s��C����Ѕ9���L�J[^:%|>�&ּ�k���?^b$�9	u��]j�qA L�{\�*�ҤH�i'� ������>�[[�`ɢV.������GVq?�CU�u]�&�*⿬b��9���3�A�.�`��A�n�������V�q�֠=*NVys�řt��K�X�1WFw���=�}��N]��?�h��A3eI�We��h���v���Wsƍ�;Z�Ƹ[�mӔa��E�|��k�����xD����?_*{��c��l��\�(��l�}(� ��=
������q���{�a���xwΨt���˅X�2�O�����s)�l�������>�g��Ye&LS#*�g���4_�"5-^$�8?"�ߤL*J!'T�<+����]�/Kq!�A�w��Z�bMp)N�נ"�,�R�\1�[Wa��㧪0C�js��߾3du����W�g���C�i��G�	�նfn��#�����{g���
�O
��w><�c��ܞ�:������.�Gy�a�����v4�Y4�eJ�]b-�O��#-�%~�o��M�M��)Q�&�P�@���A��w����H P!]�2�N'G30�}�ώ�20��,1j�5a\�)}S[�x�5>�x��hZ�ϊ��h'�3�;�A�R�%���������9yt���4C!SAut��%�v���K�S6jm����X*Fu�֢\M�@�k*�Fv�׏���޾5dt��ȍ;�Eo��W�ρB�ʘ9�f�������N�X���@gb��LK&��CR)��Z�y�/�ˡ�W�r��q����⃬����H(4wK�r��Y�+O�B�nI�T�J��'n"��������`q��k$wδ=Inj���^�|X�3�I�VCF������s��YlɉT���,Hn?�`��(\v�<�ix���d햏Ж~6�9!���p$�9ju���.���BsnɅT��ă���Z���hߝ��"Ҕ�˽YN�"�<�M�O��{������Ѕ?�z��L�1O#!|>�e���k��1�?CbV�*tn��&�4A[fI�h!�7�l��q!�}Z�=�l�����/�GP�$ɨ_\o����g��GKqX�p0�d]�4�*⿬b��Au�3��<3�I�U��� P@�ʓ񇣲�»gl�O�»p<-h���W��K�X�1W6���@�`��jt��%�a��A3e �WmP��b��⌇8���S~��'�z��h��E�����cU����7-Q|}Ӊv��ZΚ!�(��<�kU���F����Ǌy���X����ew��h	�׌��O�2�O�I����n=�ё������>�g��	J(]ohAU�Y��LW�.5Y=bK�1bX���L7JS\
)�FP�d���:�]0'�<�c����&D6yU���i6�,� �L�lWvՀ㧪0[�y|����Urz��~��W�g������S�	���``��	�����{g���5�O
��w>5�:󵑡�$�:�����%�/�V��۞9�v4�YlO�%>.e-�/<�
��}�T�o��M��L�Ks�e�P���������l����H bC'�2�Um3rH�QIW����a.��8!p&�5a\�)FnN�9�vY�f��d�쑖�h'�3�I�U�n4�?r��s�������0b^���41ZF<ui��:��̇C�!<Aj֔v����NWFu���?}0�@�jC�Wz��`壾(d��r��7�P+C��(�U���g��D�Ǖh���r��G�r���7��Fv��1dzܜ%�5�A����W���D������ޑ�Î39IwV����6i��P�=�U�uc�T�˽Zn?�%���i���ͼt��-t��=Tn���>�X�3�2�+<��◯�����u��I�N��^C$�`ڠ=!m%�<�/(���ūfF틏��1dE����p9�^b���"���BsnɅT����yꥎ=�����	��Ҕ���L5�_�Z�Y�q��{��������.���U�Qq^:%|>�e���$�8�7	-v�9|�� [�A[f;�{\�*�_��'�nA��l���؈2�e-@�`ɢVs@֭��<��GKqX�1sD�+�}��� ��x�5���<�I�U��x�n�թ����o-����"-9z�ō}��K��9Qw���I�;��F/t��?�(��IAZ�Wx���õe⌇8���S~�ѴvT�|��h֚7�m��k�����IU����7_*{��v��Ob}Ȅ\�!8ڌ<�kU���	Aw�ҘY͕�q�ͻ @�۾�ew��h	���R��E�#�e�A�e���n)��ݪ�����>�g��YJ5]_\XW����l$�_9-L?�8yr���L7JS.q<�6�~���L�#KS��j����&D6.�b׽E�^�/�"�]}vՀ㧪01�}|����&|��r��^�:g���C�i��S�r��� {��{�ř��r|ӳ�H�R
��%pz�j�����a�r�>�����"�pY�����9�v4�Y4�Xxc[�]-�x��!�]�2o��M�M�L�K=	�"�P���������l����H q{3�2�NU<$S�QIW�ώ@�'%��E!m&�Nw!�4[�9�v>�e[��z|���Υ}Z��I�U�O�B�����ٔ�}p2���41ZF<uiݗ-�]���K�A<jm��ͻ5@;h���3}+�@�b1�Fv���k��`���Sqn��x��7�Pf���ҁJ� ��D�rف���r��G�r���$!��Fm��Ch��%�<�A����W�L��N����:ȃ���ؤ3|%��ͨ)P��6�@�K�uc�T�˽Z3�c����أ��+j��-t��nn	���o�|X�3� �VW��⊲��圗YlɉT���WRaD�ڽ !C�P�q����}l틏��1d7�k�ڡ,�Bx�ݻ'-���BsnɅT�a�|�O�}������Ҕ�˽YN�"�<�C�H
n��{��
Մ����.���>�]&^'%6����k��>�?^b$�9	u��j]�IAFf\�	'�W�ĭ�}Z�=�l�����b�R�+��kUA߬��k��$'+�*YD�y]�/� �����9�~���k?�Z�N���R�n�ח����o_��ЊZ"<-h�ō}��0�%�,W4i`���@�`��F/t��D���AA_�LO��b�� v��i���Ss�./���h��E��������Q�XϽ�7-Q��v��{Il��\�3!ڌ<�kU���=
�ϐ|�Ӑj���{T���e��	���j��C�2�O�A�e���!k�
���˦���7���B`5]_.#*���]_�"'6t$$�8yr���7O)�<����]�/K(�(�j��j��]VKuU�̊"�,�R�1�=BȀy���!`O�y�Ϣ�.d��~��%����$�̆.�	���7{��	�����	���U�5|��Xv<�c󵑡�$�:������S�:�t����5�`=�s4�X>3eJ�]hk���'f�p�Y+��E�A�8�'*�,�P�������������HGXS{5�>�[.\rH�QIW����t��8S>�.K\�)4[�9�vx�	��uឝ��B'�3�I�U�O�`|ƃf������9yt���41ZFz%��-���6�p-<jm����]*Fu���uD9��H1�Fv�׏��0娣(r��x��7�Pf��,�mϜB�g��U�w����r��G�r�ϳ;vs̭[��/l��%�<�k����%� ����������ؤ39IwV�n���f`̛-�+O�B�nI�T���L?�c�����ۿ�`q��-tچTs���|R�A�X�+<��◯�����0/ɔT�3��KWZ?�`��!m%�<�4xѽɀ�&Fǋ���1dE�:#���x2�5j:�ѩHF۲��YYDɅT���K���=�l��P������MɎ�Hz!�u�!�$�:89��:�����u�-��L�J[^:l:>���k��%�Y?W�0	.��&�4A[f;�{.�>�Ϥk2�H�1�~���؈2�O@�`��-;i����ʢ:GqJ�*YD�y]�/� ���tv��^��ǥ3�@����R�����ºg_��ǠN+'h�ō}��K�*�)*Fw�ؽzQ�l��O4^��?�a��A3o�Wx��)�ЎE7����ȱ0g�د./���h��E�:��z蘀��Aq.�c���>6{��v��=l��'�U��G�N���Fw�Ґė�e���{&����ew��h	�׌��#�O�R�3�s⦉n)���������E���Y8NH"5	*���_�PN>#9�0.=���
gQy\
)�+�����$VL��j����&D6.�b׽P�8�x�1�FWvՀl���0 C�����.d��~��W�Rg����;�Y�{�¹Q��	�����{&���U�@$��%z<�j�̡�h��$�����.�:�|����D�k4�QlO�%23wC�wb-�
��f-�}/�eo��*�?�]�G{K�,�P���������z����/("ZZ�2�G<m3rH�QIW���W�ah��0S7�9aH�2[�9�v>�e��p|�����\�N�I�\�6O�Bo��{�������945���|ty'aʛ-�n죓B�p-<jm����5F;h���U{"�L�v8�lv�׏��`���Srn��x��/�KL��W�ρB�)��Q�{��i���i��G�r���@g$��Rpv��1d~܋�<�k����W�Aԕ����k�����ؤ39IwV�S_���f��+�|T�B�nI�T�˽(,�c��C�ȵ��0j��-t��=Tn���N�|O��I�V�������ڽ��O���KWZ?���@!p%�G�c��Ž}l틏؎sdX�w&���p.�Kb�ƃ.���Bsn���魍K���=�lֈ��#��ҔQ���
+5�u�!�$�:qĦ)��C������l����	RW!V>�e���6��%�?^b$�kL4��2&�4A'h�{M�?�4Ҥ:�}�=�-���ŕ2�F�JɢV.��̜s̳ZKP�Jb9�yI�4� ⿬b��,�v�ֿA�X�Y���x�n�������r-����'.<>a�ō}��K�X�J@;j���s;���S&o)��?�a��A9#�Je\��'��A2��� ���?{�ʞ/���h���m��v�����[~Y����7-Q��v��F��\�So��<�kU��� S������e���{T���e1�{t�ʌc��%��O�A�e���>R�j��������g��YJ5]-U;y*����L�de6t$�8yr�͟7AN\2�+����]�T^~Z�<�����&D6yU���_�1�y)�q*�FWvՀ���$`C�js��ߥd��~��W�kt���C���H�	���{��F�����45���A�)s��%z<�j蟑��g�'�k��Ӣ�.�:�|����9�\�Y4�X>3 �0%���$!�i\�[,��V�g�L�K{Y��P����ł�C��F����ad 6�$�d'G3rH�_�Q��]�a`��cZ&�5a\�)4g �D�k>�moǾ�����h'�3�I�U�Z�_"��>��׵ʞ�1nx���X^)O'_iݗ-��̦�KP<j��֜q��]^]9u���' 0��'�Fv���m۫`壾(d��(��J�Mfp��*�:ρB�o��D�+��f��� ��l\�r���@gb��ve��^چX��k����W�AԔ����k�����ؤ39Iw$�r�ܨnE���G�h�nI�T���
n4�c������`q�-iҞr*���>�C�3�I�V�סԯ����slɉT�E��ZH�`ڠ=!m%�y�;7���Ա}#�����p'�vj���p$�9j(���.���2%үT����2ҥ�)�v��ӝ	��N��O�ˠDn$�_��$�:q��{�������3�ە-�9>W:~V�e���k��%�M%vY�$	ܬc7�8AMo �{\�*�Ҥ:�I�=�l�����O�OI�JɢV.����(�ZKy�c7D�p/�;�0⿬b��9�7���3�R����R+�ԩ����o-��ˠZ"<_�Ő}��9�J�=WTl���@�`��F/xϖ?�aׯ3Ht4�Ww��b�� v�׭~�޸Sc���>���a��E�|��k�����2(����EV@s
����=l��\�(��$�kH�z��=
�҅���q���{T��̽|
��hn��� ��X�;�e�A�e���n)���������{�/��gUH"#pI�s���4_�"5-^$�8?"�ܤL*J4TxR�k'�
���]�/KZ�<�j��v��&Y6.�b̊"�,�R�1�,cm՝�ܲMi�j�墾.d��ݘW����X�i��S�	���=�������m��H�R
��%zz��⑼�V�/�d�����.�:�|����9�v<�Lp�n(OJ�]b-�
�� }�I�t��M�M�L�K{Y�L�-�������(��l����H PS{3�2�NU<&S�QIW�ώ@�.7��8)'i�qaV�[O&��v>�e��hҦɼ�h5��I�U�O�BFE��{�����ہu*1��1ZF<uiݗh�_Ϟ�]�V-sMf��e��O��U}lu���']�@�b1�F@$����`��i7V��l���Pf������o��D�r��1ո�r��G� �ۼ@zb֥4vg��Ca��%�<�k����SW�cԕ��ʀb������39IwV����ʍM�fG�9�E�G��Zn?�c��������}q��_��=Bg���E�|X�3�I�-k�����z��˽aeңT���KWZp�`Ǡ5v"w�<�`
���ޗ}l틏��1dE�<2�ڡb?�ju�ݩ.����6n��~�鿖a���=���Νn��s���¦sn5�_�!�$�:Ҟ{��v������.���L�J[^:%|>���k��B�M%sY�9|��&�4A[f;�{.�2�Ϥk2�K�1�y���؈2�O@�`��-7i����ʢ:GqN�*YD�y]�/� ��2��Ix�3���Ww��B���>m|�ٯn�������o-�\���Z?<J`|���q��B�r�1WFw������F2tq��B�K��A3eI�We�����v���o���S~�Ѵ/�Q����E�הp�����IU����L>,Ɔͩ&3l��\�(ڌz�G���4xa�ɺ���q���{T�a���xw��'[���AЯX�2�O�A�e���n"��ӄ����>�g��+1! _3#vQ�a��_�"5-^$�zyo�ŎeSV[�V�3���]�/KZ�}�)����n6yU�נ��R�1�F:C��Eɍ�0C�j����a6��rɪ��� �`��y�	���&��	�����)"���b�R
��v?<�~񅻡�$�:�6����m�'�m�ꀴ�v4�Y4�*E'J�]%�t��f8�~�o��M�M�7�6{D�P�"�������(��l����HR+7}'�2�FU<"D�B@}�ώ@�au֤/\p;�Ri.�8I[�0�\>�e��hធ��u'�;�2�(�Y�Y1o��{���􄝔9dt���qN}1,��:����$�S6jm����5F;h���U{"�L�p8�lv�׏��&�بUd����,�Pf��W���9���D� �� ո�r��G�4���=g��=��Ch��%�<�;�ԶW�j����׀⃬�MΣ�N9Tw$� r��f��P�=�[�nT�\�U��S(o�I������0q��-c��=Tn���7�i%�.�;�AxՖ◯����dɔT�cέaWZ?�`ڠOZyX�!�����}l틏��CV�l��iY�ju�ݩ.׳��B{9������v꾹=�lֈӝH�ҁ��Yn5�_�!�h�q$��{�������|�U��]�J6%6�&ֵ�p��%�?^b$�#u��&�v'p�Q\�*�]��I:�hJ��l�����t�GF�`ԿV?�Ԕ�g��GKqX�wsL�N�/����g��_�v���3�I�U�f��R���������E-��ˠZ"<-u���`̊C�#�L[F ~���@�`��F/tq��B�|��'R	:�LO��b�� v��q���!�ʞ/���h��E�Дk���Ⱥ4����7-Q��b�� Ξ!�ڌ<�kU���Fw���\���x���O���ew��h	Ӕ�օJ��O�A�e���3)�[��������>�g��YJ5/$:^7�{�ر"�"'$E$$�8yr���L78(Iw)�L�k��Q�=Bp�<�j����&6Mo(���P�=�R�*�FWvՀ㧪:[�wҤ��Q,B��;��B��܄0�r��S�	���{��Y�ř��o���5�R��%z<�j蟑��b�A����Џ8��|��۞9�v4�	e'�X#31� y�
��f-�T�H��M�M�7�6`s�7�P���������q����dYgP<�2�N'G3rH�QWW�ϛ[�au��8!p&�5'�T4[�B�%�e��h�劼�3�3�I�.�aT�Bo��{�������Dyi���#LAl<uiݗ-��̯	�G-4X%��m��~��3Llu���' 0�@�!r�[d�����`壾(99��x��7�fG���K�B�o��D�{�I���d���~��$!��]'\��Ch���<�k����G�Z������Q�Ƭ��39IwV�\I���%Q��M�F��DI�T�˽Z'y�k���i��쿮εH}��Tn���E�|X�H�4�Kq�䙅Ҟ��ܗslɉT���9,On?�`��OZ|X�<�`c��Ž}l틏سJr8�bq���5�5j`�ƃ.���Bsn��C����i�ނ@�l����	��Ҕ���	n(��j�[�0;��s��������5���L�J[^:%:n�t輂kƑW�-#n$�0_��&�4A[f;�+'�W���w-�fp�=�l���؈2�|S�`Ԣ$Ux�Ծ�g��GKqX�w#?�]�/�{���b��9�v�֪m �I�U�J��I�ٯn�������o-�\���Z7'h�ō}��K�*�(*Fw�ؽzQ�l��O4^��?�a��A32�*e�����*v�׭���Ss�/�z��s��E�|��k�����IU�I���7'X|}щm��=l��\�(S��<�kG�7��Fw�Ґ�Ǩ=߁� ~���ew��h	ଘl˘X�:�4�<�e���D)�������ڌE���Y-=/$?^*���_�"5-^$�Co���+?8(Mw%�"�3���]�/KZ�N�}П��.6Mh(�©	�,�R�1�F%�����BfR�f�칔.d��~��W�!Ӱ�C�(��,�Lϑ�Os��	���uȷ`M���H�R
��c*G�肑ƍV�(�s����.�:�|��۞�!�Y4�#&N~`�]b-�
��fk�t@�r��6�0�f�K{Y�7�P�����������H P! '�2�NaH`5�LIu,�oԤ@�au��8!p&�Nr!�44�k�x�~7��h�劼�ha�3�T�C�6O�Bo��{�����˹9dt�ѪO 'J<g`ƽ-����K�Z-Nt}��p��wٴU}Fu���' 0�2�vL�[�Ɵm۫`壾(d��x��7�PnU�����9���D�{�����r�RG�r���jgb��Fvψi5h��%�<�k�����;����׀����Y�ε?95Z�s݂�o��z�O�B�3c�T�˽Z,m�"��������34��8f��=Tn���E�=�3�T�G��ȗ����ǽ��T�~��0E'?�i��=!m%�<�i
���Š}���ɼ=dV�d[�ǡp$�9j���.	��¼Sbɑ]�0鿖a���=���Νn��s���¦sn5�_�!�$�|!��6�T�̥��|���8�&4-3>V>�e���k��u�.#b9�^��e*�'H@L;�{\�*���w/�}G�O�{���؈2�O@�&��BS������GKqX�1sD�)&�R��Ŀ��9�v���Pc�[�U���G�n�������|P����px$.^��}��K�X�wFj���+@�`��F/tq��B�|��IAX�[e��H�� v�׭`w�ظSc���y4���h��E��������\~N����7-Qt����=k��G�(ڌ<�k���NT8������`���QT���ew��+	�ן
�X�2�O�� ���nr�=��������{�5��Q\$Q_aa*���V�-^$�8$X���L7JSXl�]0�����|K�/�@����&py]�\��?�8�	�&1�FWvՀW���BfW�j����[k��%��W�g���C�i��S�	���O?�� ���g��{g���H�R
��6<�j��ܞ�:������.�Gy�a�����v4�Y4�>v)�]9�
��f-�T�j��M�M�D�0j$�7�Y�(��������l�̜�U `Kb�z�caz^�Q=d;�aƕj�au��8!p&�s1'�T4[�1�/�i��s+�劼�h'�3��.�aO�Bi��`������9yt���O%'F!u��P�'���K�Z-<j������<,Snn���' 0�@�b1�yd�ג��t���(d��x��7�"��J����f��_�{�����r��W�y���[Mb��Fv��C'�8�4�$�ݩ�%�ҝ(��׀⃬�ߛ�3$IeM�?���f���+O�B�nI�T�V��Z5�c��������2>��;e چXn~���uC��I�V��ȗ�����6-��~���
.�t��=!m%�<�/x��ن�`q힆ؚNE�q�ǡpm�9b���.	��޲.g��~�鿖a���=���hӀ	��zm�� �ٴBD5�_�!�$�:qu��{��\�ϥ��o�B��_�J/.HJ7�O���k��%�?^$t�(tu��.�OP&j;�rG�*�Ҥ:�}Z�m�~�����I�2j�`ɢV.��̜t̳ZKy�c7M�)F�/� ⿬b��T9�k���<3�I�U���Rpt���s�������o-��ˠZ"<b*�Ņ*���R�C,^ol���@�`��F/5@��"�s��k3eI�We��b��Ev������S~���.���h��E�|��<��ٳ�AU����eiQ��b��l��\�(ڌ<�=���Ql�Ґ���q���{T�I���1��$Fч�c��%�>�=�U�i���E�r�ӄ����>�g��YJ}.> �^��_�"5-^$�88!���H{�A{�Q��]�sKL�o�0�E�/D7dU���P.�E�j>�i8�lWvՀ㧪0C��忾a&|��e��W�g���C�i��(�t���R9��t�����{g���H� $��%z<�j蟑��$�H�g����\�G�|���9�v4�Y4�X>A[� b0�h��wP�@�E��M�M�L�K{Y�=� ����̕�J��(����D $BlH�;�d'G3rH�QIW���;�u��_)]�Hm\� /?[�9�v>�e��hGÞ���u'�H�4��O�Bo��{��Ԥ�ɹ9dt���I*pF<uiݗ-����9�IP<j��"����U}Fu���' 0�@�$a�Mv�̥��`壾(d��
��J�Mfp��*�:ρB�o��D�{��i���r��c<����@gb��Fv��CP*��%�k�9ؙ��g,�ҝ(��׀⃬����pzIjV�%���f��P�O�B�"�T�0�Zn?�c��������p��-��,)b���o�|X�3�I�V���ǯ�ä��$���J��K#9P�i��=!m%�<�ix���}���ئ9>�}�Шk�9ju�ݩ.����9dɘT�a�K���=�lֈӝ	�s �����J.�_�!�$�:q��{�j���碌U�z��L�J[^:%|>�e�����%�?,2�"#u��&�4A[f;�=�9�Ϥ~A� A�=�l���؈2�O���V3w����M��GKqX�1sD�y/�<� ��-Y��Bi�\���3�I�U����uׇ�����o-����!6A-u���ݐG�@�*}Fw���@�`��4Tb~��?���2V~c�We��b�� v���o�ȷ(fu���/���h��E�3��v�����_�l䷲J6{��v��=l���kژ'�AU���Fw���$���q���{T�0۾�ew��5	כ�T��r�2�O�A�e���<f��������X�&��PC.wu.#*����_�"5-\a�sbX���Lt 
8�1����]�fR��j����&SU�נ"�e�Z�w"�FJk�����9�@�墾.d����*�g丙8���A����{��	���h��g䳐@�)s��0s'�j蟑��$�:������.�H�m��ی0�\4�Y4�X>3e8�J-�mɯ<�T�t��M�M�L�K{Y�L�-���������{����H P! '�2�Da3oH�B(�W���ic��LQI�<zv�)4[�9�v>�5f��h���ΥzZ�3�@��O�Bo��{�������9��1ZF<uiݗ-�Kз�6�G-Nt}��G����NWFuΎ�4}0�@�'�](v�׏��`��xƋeقL�-}(��W�ρB�o��W�{�����6��A\�r���@gb��FK&��^~��%�<�k����%� ���ﰈb�����ѿ9IwV����i��P�(�0�4�T�ЗZn?�c����芧�`l��V`q��=Tn���E�|*�'�I�Vwm�����ǽslɉ���W?�i��%\v�<�ix��Ž</�����*No�q�ǡp$�9/9Ϙ�u>-���BsnɅT�a�|�O�}��Ĕ��Ҕ�˽YN�"�<�C�H
n��{��
Մ����.���>�\&^'%6����k��>�?^b$�9	u��j]�IAFf\�	'�W�ǭ�}Z�=�l����*�O@�h��GS����g��GKqX�1sN�)]�/�AQ���'J�V1�z���z\�@����R�(�������%�w�Z656B�ō}��K�X�a,Pow�нzT�{��F/t��?�a��:&I�Wj��yۯ� v�׭���(ju�̴vT�|��h��E�|��k���ȼ4H�l��,Q��v��=*��N�(��)�p���Fw�Ґ|�ސq���,�^���~]��h	�׌˅�2�R�V�O���n)�������ʃ>�gӻ+1$ S.11���_�"5-,u2�8dr��� [Qy\
)�+����&�RKZ�G�����&D6yU���_�1�y)�q*�FWvՀ㧪BfP�j��٧S-��~��W�g�W�i��!����7{��	�����4%���@�XJ��,G�󵑡�$�:����.�/�V��۞9�v4�4�X>3eJ�]'a�J
��L-�T�o����D�ZwY�u�P�
ǭ����(��l����
z! '�2�ur9S�QIW�S��@�tc��8!p&�5(�!uV�$�v,�eF��h�劼�hn�3�;�A�R�Br	���𒯤��9yt���41Z4G`݊-���6�Z?5@m����NW4hc���'g8�;�=�Tm�׏��`壾(h��x��P�"��W�ԫB�o��D�{�����o��P����"0��J��,la��%�<�k�����j���ﰈb�����ѿ9IwV���� B̙-�O�9�R�T�˽Zn?�c�����쿵g��t��=Tn����n%�.�;�ExՖ◯���ǽ��T���SUo�Jڠ=!m%�<�i>��ؽhwǋ���1dE�q���$�9}S����BsnɅT���#���5�#��ٔ{��u���˽Yn5�_�b�$�:cd��{��������k�T���`[^:%|>�e�����%�?9jV�+ty��=�4A[f;�{\�*�e��'�R�F���ѓ�O@�`ɢV.f~����gֻ50`%�1gM�S]�/� ⿬by��y9�v�ުm"�E�@���R�n�������r-����'.<;a�ō}��K�X�1] Bw�Ђ@���k1Q��3���.@lR�We��b�� 0���o�Ȣ[s���:���h��E�|��k����IU�e�۹-Q��v��=_<Ν!�5`��A�AU���Fw���^�Ӑq�Ļ @�۾�ew��h	���A��%�/�=�V�~���n)���������C�z��"\HFu.#*���-�3%P^$�o6 �ğ,`S\
)�+����]�2KA�<�j����&DDA�ʠu*�W�v^�8�lWvՀ㧪0o8�����;��~��W�g���8���S�r���7{��	�����	���U� qι>P<�j蟑��$�u������|�0����v4�Y4�X>r&	�@b>�% ��f-�T�EE��M�M�L�K>�r��(������¨>�Ş�^1\!Oe�2�Ffp{A�{cW�ώ�Ku��8!pd�p �4[�z�%{�t��B�劼�!a�;�
�U�O�K4��{������Ka���4VR4Ggї?����K�Z-<������<,Wny���<*0�@�b1�Fp�����h�دUh��c��7�Pf��%���_���?������r��G�r�Yb��!��>h��>�<�k����%� ���ﰈb�����ѿ9IwV����6���J
�
�/��ǽ.S��������`q�Veq��=3fj���I�oQ��I�V����ԅ|����}��O���KWZy���=<mW�%�rR��Ž}l��ߣ�LdX�
���Z$�9ju�ݩhD|����B��O�鿖a���{���Ν{��u���˽Yn5��3�$�:Ҟ`��������\���L�JS	uw87�5���k��%�?^$t�2u��2&�4A[f;�	'�8�Ϥk2�H�1�x���؈2�O@���+.	腫�|��GKqX�1s6�m �2�ra�·H��9�v���N�T�'���I(�n����נo0�~�ڲ'9-h�ō}̢	�E�9 	@3���s;���F/t��?� ��A.eZ�}O��b� 3���W���S~�Ѵj�S�`��I�3��k����@
N�4���7-QS,��v��=.���3!ڌ<�*���^]�Ґ�Ǥ7�̈8���lw��B	�׌˅X�I�2�\����T�������>�g��5@_cbOo�T��[�79-*mH�Kpi���L7JS\
o�m:����U�TZ~V�/�q����&D6y�d��?�^�/�&1�FWvՀX�ܹM^�z����.d��~���ku���C���H�	���{��{�Ǚ��s0��A�$��%z<�j�����9�/�U�����.�H�i��ƞ(�n4�Y&�X53w_�FH-�
��f-�MT�g���M�E�0m$��P�������A��l����H P! '�2�t3)b�QIW�ώ�3:��.0|&�wm\�!uV�0�\�e��h\�ϊ��h'�q���6O�B,��>�����9yt�ّr1R6i��-����a�Z-<j֙q��N0Na���'9�j�b1�Fv�����޾5y��k��7�zL��W�ρB�ʕ9�f�������N�X���@gb��F��>u��-�G��