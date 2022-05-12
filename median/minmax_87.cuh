#ifndef MINMAX_87_CUH
#define MINMAX_87_CUH

namespace
{
  template <typename T>
__device__
void minmax_87(T *array, int size)
{
  switch (size) {
  case 3:
    minmax3(array + 0, array + 1, array + 2);
    return;
  case 4:
    minmax4(array + 0, array + 1, array + 2, array + 3);
    return;
  case 5:
    minmax5(array + 0, array + 1, array + 2, array + 3, array + 4);
    return;
  case 6:
    minmax6(array + 0, array + 1, array + 2, array + 3, array + 4, array + 5);
    return;
  case 7:
    minmax7(array + 0, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6);
    return;
  case 8:
    minmax8(array + 0, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7);
    return;
  case 9:
    minmax9(array + 0, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8);
    return;
  case 10:
    minmax10(array + 0, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9);
    return;
  case 11:
    minmax11(array + 0, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10);
    return;
  case 12:
    minmax12(array + 0, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11);
    return;
  case 13:
    minmax13(array + 0, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12);
    return;
  case 14:
    minmax14(array + 0, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13);
    return;
  case 15:
    minmax15(array + 0, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14);
    return;
  case 16:
    minmax16(array + 0, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15);
    return;
  case 17:
    minmax17(array + 0, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16);
    return;
  case 18:
    minmax18(array + 0, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17);
    return;
  case 19:
    minmax19(array + 0, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18);
    return;
  case 20:
    minmax20(array + 0, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19);
    return;
  case 21:
    minmax21(array + 0, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20);
    return;
  case 22:
    minmax22(array + 0, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21);
    return;
  case 23:
    minmax23(array + 0, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22);
    return;
  case 24:
    minmax24(array + 0, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23);
    return;
  case 25:
    minmax25(array + 0, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24);
    return;
  case 26:
    minmax26(array + 0, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25);
    return;
  case 27:
    minmax27(array + 0, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26);
    return;
  case 28:
    minmax28(array + 0, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27);
    return;
  case 29:
    minmax29(array + 0, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28);
    return;
  case 30:
    minmax30(array + 0, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29);
    return;
  case 31:
    minmax31(array + 0, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29, array + 30);
    return;
  case 32:
    minmax32(array + 0, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29, array + 30, array + 31);
    return;
  case 33:
    minmax33(array + 0, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29, array + 30, array + 31, array + 32);
    return;
  case 34:
    minmax34(array + 0, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29, array + 30, array + 31, array + 32, array + 33);
    return;
  case 35:
    minmax35(array + 0, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29, array + 30, array + 31, array + 32, array + 33, array + 34);
    return;
  case 36:
    minmax36(array + 0, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29, array + 30, array + 31, array + 32, array + 33, array + 34, array + 35);
    return;
  case 37:
    minmax37(array + 0, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29, array + 30, array + 31, array + 32, array + 33, array + 34, array + 35, array + 36);
    return;
  case 38:
    minmax38(array + 0, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29, array + 30, array + 31, array + 32, array + 33, array + 34, array + 35, array + 36, array + 37);
    return;
  case 39:
    minmax39(array + 0, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29, array + 30, array + 31, array + 32, array + 33, array + 34, array + 35, array + 36, array + 37, array + 38);
    return;
  case 40:
    minmax40(array + 0, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29, array + 30, array + 31, array + 32, array + 33, array + 34, array + 35, array + 36, array + 37, array + 38, array + 39);
    return;
  case 41:
    minmax41(array + 0, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29, array + 30, array + 31, array + 32, array + 33, array + 34, array + 35, array + 36, array + 37, array + 38, array + 39, array + 40);
    return;
  case 42:
    minmax42(array + 0, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29, array + 30, array + 31, array + 32, array + 33, array + 34, array + 35, array + 36, array + 37, array + 38, array + 39, array + 40, array + 41);
    return;
  case 43:
    minmax43(array + 0, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29, array + 30, array + 31, array + 32, array + 33, array + 34, array + 35, array + 36, array + 37, array + 38, array + 39, array + 40, array + 41, array + 42);
    return;
  case 44:
    minmax44(array + 0, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29, array + 30, array + 31, array + 32, array + 33, array + 34, array + 35, array + 36, array + 37, array + 38, array + 39, array + 40, array + 41, array + 42, array + 43);
    return;
  case 45:
    minmax45(array + 0, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29, array + 30, array + 31, array + 32, array + 33, array + 34, array + 35, array + 36, array + 37, array + 38, array + 39, array + 40, array + 41, array + 42, array + 43, array + 44);
    return;
  case 46:
    minmax46(array + 0, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29, array + 30, array + 31, array + 32, array + 33, array + 34, array + 35, array + 36, array + 37, array + 38, array + 39, array + 40, array + 41, array + 42, array + 43, array + 44, array + 45);
    return;
  case 47:
    minmax47(array + 0, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29, array + 30, array + 31, array + 32, array + 33, array + 34, array + 35, array + 36, array + 37, array + 38, array + 39, array + 40, array + 41, array + 42, array + 43, array + 44, array + 45, array + 46);
    return;
  case 48:
    minmax48(array + 0, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29, array + 30, array + 31, array + 32, array + 33, array + 34, array + 35, array + 36, array + 37, array + 38, array + 39, array + 40, array + 41, array + 42, array + 43, array + 44, array + 45, array + 46, array + 47);
    return;
  case 49:
    minmax49(array + 0, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29, array + 30, array + 31, array + 32, array + 33, array + 34, array + 35, array + 36, array + 37, array + 38, array + 39, array + 40, array + 41, array + 42, array + 43, array + 44, array + 45, array + 46, array + 47, array + 48);
    return;
  case 50:
    minmax50(array + 0, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29, array + 30, array + 31, array + 32, array + 33, array + 34, array + 35, array + 36, array + 37, array + 38, array + 39, array + 40, array + 41, array + 42, array + 43, array + 44, array + 45, array + 46, array + 47, array + 48, array + 49);
    return;
  case 51:
    minmax51(array + 0, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29, array + 30, array + 31, array + 32, array + 33, array + 34, array + 35, array + 36, array + 37, array + 38, array + 39, array + 40, array + 41, array + 42, array + 43, array + 44, array + 45, array + 46, array + 47, array + 48, array + 49, array + 50);
    return;
  case 52:
    minmax52(array + 0, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29, array + 30, array + 31, array + 32, array + 33, array + 34, array + 35, array + 36, array + 37, array + 38, array + 39, array + 40, array + 41, array + 42, array + 43, array + 44, array + 45, array + 46, array + 47, array + 48, array + 49, array + 50, array + 51);
    return;
  case 53:
    minmax53(array + 0, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29, array + 30, array + 31, array + 32, array + 33, array + 34, array + 35, array + 36, array + 37, array + 38, array + 39, array + 40, array + 41, array + 42, array + 43, array + 44, array + 45, array + 46, array + 47, array + 48, array + 49, array + 50, array + 51, array + 52);
    return;
  case 54:
    minmax54(array + 0, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29, array + 30, array + 31, array + 32, array + 33, array + 34, array + 35, array + 36, array + 37, array + 38, array + 39, array + 40, array + 41, array + 42, array + 43, array + 44, array + 45, array + 46, array + 47, array + 48, array + 49, array + 50, array + 51, array + 52, array + 53);
    return;
  case 55:
    minmax55(array + 0, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29, array + 30, array + 31, array + 32, array + 33, array + 34, array + 35, array + 36, array + 37, array + 38, array + 39, array + 40, array + 41, array + 42, array + 43, array + 44, array + 45, array + 46, array + 47, array + 48, array + 49, array + 50, array + 51, array + 52, array + 53, array + 54);
    return;
  case 56:
    minmax56(array + 0, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29, array + 30, array + 31, array + 32, array + 33, array + 34, array + 35, array + 36, array + 37, array + 38, array + 39, array + 40, array + 41, array + 42, array + 43, array + 44, array + 45, array + 46, array + 47, array + 48, array + 49, array + 50, array + 51, array + 52, array + 53, array + 54, array + 55);
    return;
  case 57:
    minmax57(array + 0, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29, array + 30, array + 31, array + 32, array + 33, array + 34, array + 35, array + 36, array + 37, array + 38, array + 39, array + 40, array + 41, array + 42, array + 43, array + 44, array + 45, array + 46, array + 47, array + 48, array + 49, array + 50, array + 51, array + 52, array + 53, array + 54, array + 55, array + 56);
    return;
  case 58:
    minmax58(array + 0, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29, array + 30, array + 31, array + 32, array + 33, array + 34, array + 35, array + 36, array + 37, array + 38, array + 39, array + 40, array + 41, array + 42, array + 43, array + 44, array + 45, array + 46, array + 47, array + 48, array + 49, array + 50, array + 51, array + 52, array + 53, array + 54, array + 55, array + 56, array + 57);
    return;
  case 59:
    minmax59(array + 0, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29, array + 30, array + 31, array + 32, array + 33, array + 34, array + 35, array + 36, array + 37, array + 38, array + 39, array + 40, array + 41, array + 42, array + 43, array + 44, array + 45, array + 46, array + 47, array + 48, array + 49, array + 50, array + 51, array + 52, array + 53, array + 54, array + 55, array + 56, array + 57, array + 58);
    return;
  case 60:
    minmax60(array + 0, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29, array + 30, array + 31, array + 32, array + 33, array + 34, array + 35, array + 36, array + 37, array + 38, array + 39, array + 40, array + 41, array + 42, array + 43, array + 44, array + 45, array + 46, array + 47, array + 48, array + 49, array + 50, array + 51, array + 52, array + 53, array + 54, array + 55, array + 56, array + 57, array + 58, array + 59);
    return;
  case 61:
    minmax61(array + 0, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29, array + 30, array + 31, array + 32, array + 33, array + 34, array + 35, array + 36, array + 37, array + 38, array + 39, array + 40, array + 41, array + 42, array + 43, array + 44, array + 45, array + 46, array + 47, array + 48, array + 49, array + 50, array + 51, array + 52, array + 53, array + 54, array + 55, array + 56, array + 57, array + 58, array + 59, array + 60);
    return;
  case 62:
    minmax62(array + 0, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29, array + 30, array + 31, array + 32, array + 33, array + 34, array + 35, array + 36, array + 37, array + 38, array + 39, array + 40, array + 41, array + 42, array + 43, array + 44, array + 45, array + 46, array + 47, array + 48, array + 49, array + 50, array + 51, array + 52, array + 53, array + 54, array + 55, array + 56, array + 57, array + 58, array + 59, array + 60, array + 61);
    return;
  case 63:
    minmax63(array + 0, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29, array + 30, array + 31, array + 32, array + 33, array + 34, array + 35, array + 36, array + 37, array + 38, array + 39, array + 40, array + 41, array + 42, array + 43, array + 44, array + 45, array + 46, array + 47, array + 48, array + 49, array + 50, array + 51, array + 52, array + 53, array + 54, array + 55, array + 56, array + 57, array + 58, array + 59, array + 60, array + 61, array + 62);
    return;
  case 64:
    minmax64(array + 0, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29, array + 30, array + 31, array + 32, array + 33, array + 34, array + 35, array + 36, array + 37, array + 38, array + 39, array + 40, array + 41, array + 42, array + 43, array + 44, array + 45, array + 46, array + 47, array + 48, array + 49, array + 50, array + 51, array + 52, array + 53, array + 54, array + 55, array + 56, array + 57, array + 58, array + 59, array + 60, array + 61, array + 62, array + 63);
    return;
  case 65:
    minmax65(array + 0, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29, array + 30, array + 31, array + 32, array + 33, array + 34, array + 35, array + 36, array + 37, array + 38, array + 39, array + 40, array + 41, array + 42, array + 43, array + 44, array + 45, array + 46, array + 47, array + 48, array + 49, array + 50, array + 51, array + 52, array + 53, array + 54, array + 55, array + 56, array + 57, array + 58, array + 59, array + 60, array + 61, array + 62, array + 63, array + 64);
    return;
  case 66:
    minmax66(array + 0, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29, array + 30, array + 31, array + 32, array + 33, array + 34, array + 35, array + 36, array + 37, array + 38, array + 39, array + 40, array + 41, array + 42, array + 43, array + 44, array + 45, array + 46, array + 47, array + 48, array + 49, array + 50, array + 51, array + 52, array + 53, array + 54, array + 55, array + 56, array + 57, array + 58, array + 59, array + 60, array + 61, array + 62, array + 63, array + 64, array + 65);
    return;
  case 67:
    minmax67(array + 0, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29, array + 30, array + 31, array + 32, array + 33, array + 34, array + 35, array + 36, array + 37, array + 38, array + 39, array + 40, array + 41, array + 42, array + 43, array + 44, array + 45, array + 46, array + 47, array + 48, array + 49, array + 50, array + 51, array + 52, array + 53, array + 54, array + 55, array + 56, array + 57, array + 58, array + 59, array + 60, array + 61, array + 62, array + 63, array + 64, array + 65, array + 66);
    return;
  case 68:
    minmax68(array + 0, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29, array + 30, array + 31, array + 32, array + 33, array + 34, array + 35, array + 36, array + 37, array + 38, array + 39, array + 40, array + 41, array + 42, array + 43, array + 44, array + 45, array + 46, array + 47, array + 48, array + 49, array + 50, array + 51, array + 52, array + 53, array + 54, array + 55, array + 56, array + 57, array + 58, array + 59, array + 60, array + 61, array + 62, array + 63, array + 64, array + 65, array + 66, array + 67);
    return;
  case 69:
    minmax69(array + 0, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29, array + 30, array + 31, array + 32, array + 33, array + 34, array + 35, array + 36, array + 37, array + 38, array + 39, array + 40, array + 41, array + 42, array + 43, array + 44, array + 45, array + 46, array + 47, array + 48, array + 49, array + 50, array + 51, array + 52, array + 53, array + 54, array + 55, array + 56, array + 57, array + 58, array + 59, array + 60, array + 61, array + 62, array + 63, array + 64, array + 65, array + 66, array + 67, array + 68);
    return;
  case 70:
    minmax70(array + 0, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29, array + 30, array + 31, array + 32, array + 33, array + 34, array + 35, array + 36, array + 37, array + 38, array + 39, array + 40, array + 41, array + 42, array + 43, array + 44, array + 45, array + 46, array + 47, array + 48, array + 49, array + 50, array + 51, array + 52, array + 53, array + 54, array + 55, array + 56, array + 57, array + 58, array + 59, array + 60, array + 61, array + 62, array + 63, array + 64, array + 65, array + 66, array + 67, array + 68, array + 69);
    return;
  case 71:
    minmax71(array + 0, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29, array + 30, array + 31, array + 32, array + 33, array + 34, array + 35, array + 36, array + 37, array + 38, array + 39, array + 40, array + 41, array + 42, array + 43, array + 44, array + 45, array + 46, array + 47, array + 48, array + 49, array + 50, array + 51, array + 52, array + 53, array + 54, array + 55, array + 56, array + 57, array + 58, array + 59, array + 60, array + 61, array + 62, array + 63, array + 64, array + 65, array + 66, array + 67, array + 68, array + 69, array + 70);
    return;
  case 72:
    minmax72(array + 0, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29, array + 30, array + 31, array + 32, array + 33, array + 34, array + 35, array + 36, array + 37, array + 38, array + 39, array + 40, array + 41, array + 42, array + 43, array + 44, array + 45, array + 46, array + 47, array + 48, array + 49, array + 50, array + 51, array + 52, array + 53, array + 54, array + 55, array + 56, array + 57, array + 58, array + 59, array + 60, array + 61, array + 62, array + 63, array + 64, array + 65, array + 66, array + 67, array + 68, array + 69, array + 70, array + 71);
    return;
  case 73:
    minmax73(array + 0, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29, array + 30, array + 31, array + 32, array + 33, array + 34, array + 35, array + 36, array + 37, array + 38, array + 39, array + 40, array + 41, array + 42, array + 43, array + 44, array + 45, array + 46, array + 47, array + 48, array + 49, array + 50, array + 51, array + 52, array + 53, array + 54, array + 55, array + 56, array + 57, array + 58, array + 59, array + 60, array + 61, array + 62, array + 63, array + 64, array + 65, array + 66, array + 67, array + 68, array + 69, array + 70, array + 71, array + 72);
    return;
  case 74:
    minmax74(array + 0, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29, array + 30, array + 31, array + 32, array + 33, array + 34, array + 35, array + 36, array + 37, array + 38, array + 39, array + 40, array + 41, array + 42, array + 43, array + 44, array + 45, array + 46, array + 47, array + 48, array + 49, array + 50, array + 51, array + 52, array + 53, array + 54, array + 55, array + 56, array + 57, array + 58, array + 59, array + 60, array + 61, array + 62, array + 63, array + 64, array + 65, array + 66, array + 67, array + 68, array + 69, array + 70, array + 71, array + 72, array + 73);
    return;
  case 75:
    minmax75(array + 0, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29, array + 30, array + 31, array + 32, array + 33, array + 34, array + 35, array + 36, array + 37, array + 38, array + 39, array + 40, array + 41, array + 42, array + 43, array + 44, array + 45, array + 46, array + 47, array + 48, array + 49, array + 50, array + 51, array + 52, array + 53, array + 54, array + 55, array + 56, array + 57, array + 58, array + 59, array + 60, array + 61, array + 62, array + 63, array + 64, array + 65, array + 66, array + 67, array + 68, array + 69, array + 70, array + 71, array + 72, array + 73, array + 74);
    return;
  case 76:
    minmax76(array + 0, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29, array + 30, array + 31, array + 32, array + 33, array + 34, array + 35, array + 36, array + 37, array + 38, array + 39, array + 40, array + 41, array + 42, array + 43, array + 44, array + 45, array + 46, array + 47, array + 48, array + 49, array + 50, array + 51, array + 52, array + 53, array + 54, array + 55, array + 56, array + 57, array + 58, array + 59, array + 60, array + 61, array + 62, array + 63, array + 64, array + 65, array + 66, array + 67, array + 68, array + 69, array + 70, array + 71, array + 72, array + 73, array + 74, array + 75);
    return;
  case 77:
    minmax77(array + 0, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29, array + 30, array + 31, array + 32, array + 33, array + 34, array + 35, array + 36, array + 37, array + 38, array + 39, array + 40, array + 41, array + 42, array + 43, array + 44, array + 45, array + 46, array + 47, array + 48, array + 49, array + 50, array + 51, array + 52, array + 53, array + 54, array + 55, array + 56, array + 57, array + 58, array + 59, array + 60, array + 61, array + 62, array + 63, array + 64, array + 65, array + 66, array + 67, array + 68, array + 69, array + 70, array + 71, array + 72, array + 73, array + 74, array + 75, array + 76);
    return;
  case 78:
    minmax78(array + 0, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29, array + 30, array + 31, array + 32, array + 33, array + 34, array + 35, array + 36, array + 37, array + 38, array + 39, array + 40, array + 41, array + 42, array + 43, array + 44, array + 45, array + 46, array + 47, array + 48, array + 49, array + 50, array + 51, array + 52, array + 53, array + 54, array + 55, array + 56, array + 57, array + 58, array + 59, array + 60, array + 61, array + 62, array + 63, array + 64, array + 65, array + 66, array + 67, array + 68, array + 69, array + 70, array + 71, array + 72, array + 73, array + 74, array + 75, array + 76, array + 77);
    return;
  case 79:
    minmax79(array + 0, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29, array + 30, array + 31, array + 32, array + 33, array + 34, array + 35, array + 36, array + 37, array + 38, array + 39, array + 40, array + 41, array + 42, array + 43, array + 44, array + 45, array + 46, array + 47, array + 48, array + 49, array + 50, array + 51, array + 52, array + 53, array + 54, array + 55, array + 56, array + 57, array + 58, array + 59, array + 60, array + 61, array + 62, array + 63, array + 64, array + 65, array + 66, array + 67, array + 68, array + 69, array + 70, array + 71, array + 72, array + 73, array + 74, array + 75, array + 76, array + 77, array + 78);
    return;
  case 80:
    minmax80(array + 0, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29, array + 30, array + 31, array + 32, array + 33, array + 34, array + 35, array + 36, array + 37, array + 38, array + 39, array + 40, array + 41, array + 42, array + 43, array + 44, array + 45, array + 46, array + 47, array + 48, array + 49, array + 50, array + 51, array + 52, array + 53, array + 54, array + 55, array + 56, array + 57, array + 58, array + 59, array + 60, array + 61, array + 62, array + 63, array + 64, array + 65, array + 66, array + 67, array + 68, array + 69, array + 70, array + 71, array + 72, array + 73, array + 74, array + 75, array + 76, array + 77, array + 78, array + 79);
    return;
  case 81:
    minmax81(array + 0, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29, array + 30, array + 31, array + 32, array + 33, array + 34, array + 35, array + 36, array + 37, array + 38, array + 39, array + 40, array + 41, array + 42, array + 43, array + 44, array + 45, array + 46, array + 47, array + 48, array + 49, array + 50, array + 51, array + 52, array + 53, array + 54, array + 55, array + 56, array + 57, array + 58, array + 59, array + 60, array + 61, array + 62, array + 63, array + 64, array + 65, array + 66, array + 67, array + 68, array + 69, array + 70, array + 71, array + 72, array + 73, array + 74, array + 75, array + 76, array + 77, array + 78, array + 79, array + 80);
    return;
  case 82:
    minmax82(array + 0, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29, array + 30, array + 31, array + 32, array + 33, array + 34, array + 35, array + 36, array + 37, array + 38, array + 39, array + 40, array + 41, array + 42, array + 43, array + 44, array + 45, array + 46, array + 47, array + 48, array + 49, array + 50, array + 51, array + 52, array + 53, array + 54, array + 55, array + 56, array + 57, array + 58, array + 59, array + 60, array + 61, array + 62, array + 63, array + 64, array + 65, array + 66, array + 67, array + 68, array + 69, array + 70, array + 71, array + 72, array + 73, array + 74, array + 75, array + 76, array + 77, array + 78, array + 79, array + 80, array + 81);
    return;
  case 83:
    minmax83(array + 0, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29, array + 30, array + 31, array + 32, array + 33, array + 34, array + 35, array + 36, array + 37, array + 38, array + 39, array + 40, array + 41, array + 42, array + 43, array + 44, array + 45, array + 46, array + 47, array + 48, array + 49, array + 50, array + 51, array + 52, array + 53, array + 54, array + 55, array + 56, array + 57, array + 58, array + 59, array + 60, array + 61, array + 62, array + 63, array + 64, array + 65, array + 66, array + 67, array + 68, array + 69, array + 70, array + 71, array + 72, array + 73, array + 74, array + 75, array + 76, array + 77, array + 78, array + 79, array + 80, array + 81, array + 82);
    return;
  case 84:
    minmax84(array + 0, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29, array + 30, array + 31, array + 32, array + 33, array + 34, array + 35, array + 36, array + 37, array + 38, array + 39, array + 40, array + 41, array + 42, array + 43, array + 44, array + 45, array + 46, array + 47, array + 48, array + 49, array + 50, array + 51, array + 52, array + 53, array + 54, array + 55, array + 56, array + 57, array + 58, array + 59, array + 60, array + 61, array + 62, array + 63, array + 64, array + 65, array + 66, array + 67, array + 68, array + 69, array + 70, array + 71, array + 72, array + 73, array + 74, array + 75, array + 76, array + 77, array + 78, array + 79, array + 80, array + 81, array + 82, array + 83);
    return;
  case 85:
    minmax85(array + 0, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29, array + 30, array + 31, array + 32, array + 33, array + 34, array + 35, array + 36, array + 37, array + 38, array + 39, array + 40, array + 41, array + 42, array + 43, array + 44, array + 45, array + 46, array + 47, array + 48, array + 49, array + 50, array + 51, array + 52, array + 53, array + 54, array + 55, array + 56, array + 57, array + 58, array + 59, array + 60, array + 61, array + 62, array + 63, array + 64, array + 65, array + 66, array + 67, array + 68, array + 69, array + 70, array + 71, array + 72, array + 73, array + 74, array + 75, array + 76, array + 77, array + 78, array + 79, array + 80, array + 81, array + 82, array + 83, array + 84);
    return;
  case 86:
    minmax86(array + 0, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29, array + 30, array + 31, array + 32, array + 33, array + 34, array + 35, array + 36, array + 37, array + 38, array + 39, array + 40, array + 41, array + 42, array + 43, array + 44, array + 45, array + 46, array + 47, array + 48, array + 49, array + 50, array + 51, array + 52, array + 53, array + 54, array + 55, array + 56, array + 57, array + 58, array + 59, array + 60, array + 61, array + 62, array + 63, array + 64, array + 65, array + 66, array + 67, array + 68, array + 69, array + 70, array + 71, array + 72, array + 73, array + 74, array + 75, array + 76, array + 77, array + 78, array + 79, array + 80, array + 81, array + 82, array + 83, array + 84, array + 85);
    return;
  case 87:
    minmax87(array + 0, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29, array + 30, array + 31, array + 32, array + 33, array + 34, array + 35, array + 36, array + 37, array + 38, array + 39, array + 40, array + 41, array + 42, array + 43, array + 44, array + 45, array + 46, array + 47, array + 48, array + 49, array + 50, array + 51, array + 52, array + 53, array + 54, array + 55, array + 56, array + 57, array + 58, array + 59, array + 60, array + 61, array + 62, array + 63, array + 64, array + 65, array + 66, array + 67, array + 68, array + 69, array + 70, array + 71, array + 72, array + 73, array + 74, array + 75, array + 76, array + 77, array + 78, array + 79, array + 80, array + 81, array + 82, array + 83, array + 84, array + 85, array + 86);
    return;
  }
}
} // unnamed namespace

#endif /* MINMAX_87_CUH */