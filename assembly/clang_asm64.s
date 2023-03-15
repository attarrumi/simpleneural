#include "textflag.h"


TEXT clang_one(SB), NOSPLIT, $0

MOVQ fn+0(FP), DI
MOVQ x+8(FP), X0

CALL DI

MOVSD X0, ret+16(FP)
RET


TEXT clang_two(SB), NOSPLIT, $0

MOVQ fn+0(FP), DI
MOVQ x+8(FP), X0
MOVQ y+16(FP), X1

CALL DI

MOVSD X0, ret+24(FP)
RET

TEXT clang_three(SB), NOSPLIT, $0

MOVQ fn+0(FP), DI
MOVQ x+8(FP), X0
MOVQ y+16(FP), X1
MOVQ z+24(FP), X2

CALL DI

MOVSD X0, ret+32(FP)
RET
