#ifndef MINMAX_64_CUH
#define MINMAX_64_CUH

namespace
{
  template <typename T>
__device__
void minmax_64(T *array, int size)
{
  switch (size) {
  case 2:
    mm2(array + 0, array + 1);
    return;
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
    minmax16(array, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15);
    return;
  case 17:
    minmax17(array, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16);
    return;
  case 18:
    minmax18(array, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17);
    return;
  case 19:
    minmax19(array, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18);
    return;
  case 20:
    minmax20(array, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19);
    return;
  case 21:
    minmax21(array, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20);
    return;
  case 22:
    minmax22(array, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21);
    return;
  case 23:
    minmax23(array, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22);
    return;
  case 24:
    minmax24(array, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23);
    return;
  case 25:
    minmax25(array, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24);
    return;
  case 26:
    minmax26(array, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25);
    return;
  case 27:
    minmax27(array, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26);
    return;
  case 28:
    minmax28(array, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27);
    return;
  case 29:
    minmax29(array, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28);
    return;
  case 30:
    minmax30(array, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29);
    return;
  case 31:
    minmax31(array, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29, array + 30);
    return;
  case 32:
    minmax32(array, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29, array + 30, array + 31);
    return;
  case 33:
    minmax33(array, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29, array + 30, array + 31, array + 32);
    return;
  case 34:
    minmax34(array, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29, array + 30, array + 31, array + 32, array + 33);
    return;
  case 35:
    minmax35(array, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29, array + 30, array + 31, array + 32, array + 33, array + 34);
    return;
  case 36:
    minmax36(array, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29, array + 30, array + 31, array + 32, array + 33, array + 34, array + 35);
    return;
  case 37:
    minmax37(array, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29, array + 30, array + 31, array + 32, array + 33, array + 34, array + 35, array + 36);
    return;
  case 38:
    minmax38(array, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29, array + 30, array + 31, array + 32, array + 33, array + 34, array + 35, array + 36, array + 37);
    return;
  case 39:
    minmax39(array, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29, array + 30, array + 31, array + 32, array + 33, array + 34, array + 35, array + 36, array + 37, array + 38);
    return;
  case 40:
    minmax40(array, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29, array + 30, array + 31, array + 32, array + 33, array + 34, array + 35, array + 36, array + 37, array + 38, array + 39);
    return;
  case 41:
    minmax41(array, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29, array + 30, array + 31, array + 32, array + 33, array + 34, array + 35, array + 36, array + 37, array + 38, array + 39, array + 40);
    return;
  case 42:
    minmax42(array, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29, array + 30, array + 31, array + 32, array + 33, array + 34, array + 35, array + 36, array + 37, array + 38, array + 39, array + 40, array + 41);
    return;
  case 43:
    minmax43(array, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29, array + 30, array + 31, array + 32, array + 33, array + 34, array + 35, array + 36, array + 37, array + 38, array + 39, array + 40, array + 41, array + 42);
    return;
  case 44:
    minmax44(array, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29, array + 30, array + 31, array + 32, array + 33, array + 34, array + 35, array + 36, array + 37, array + 38, array + 39, array + 40, array + 41, array + 42, array + 43);
    return;
  case 45:
    minmax45(array, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29, array + 30, array + 31, array + 32, array + 33, array + 34, array + 35, array + 36, array + 37, array + 38, array + 39, array + 40, array + 41, array + 42, array + 43, array + 44);
    return;
  case 46:
    minmax46(array, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29, array + 30, array + 31, array + 32, array + 33, array + 34, array + 35, array + 36, array + 37, array + 38, array + 39, array + 40, array + 41, array + 42, array + 43, array + 44, array + 45);
    return;
  case 47:
    minmax47(array, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29, array + 30, array + 31, array + 32, array + 33, array + 34, array + 35, array + 36, array + 37, array + 38, array + 39, array + 40, array + 41, array + 42, array + 43, array + 44, array + 45, array + 46);
    return;
  case 48:
    minmax48(array, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29, array + 30, array + 31, array + 32, array + 33, array + 34, array + 35, array + 36, array + 37, array + 38, array + 39, array + 40, array + 41, array + 42, array + 43, array + 44, array + 45, array + 46, array + 47);
    return;
  case 49:
    minmax49(array, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29, array + 30, array + 31, array + 32, array + 33, array + 34, array + 35, array + 36, array + 37, array + 38, array + 39, array + 40, array + 41, array + 42, array + 43, array + 44, array + 45, array + 46, array + 47, array + 48);
    return;
  case 50:
    minmax50(array, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29, array + 30, array + 31, array + 32, array + 33, array + 34, array + 35, array + 36, array + 37, array + 38, array + 39, array + 40, array + 41, array + 42, array + 43, array + 44, array + 45, array + 46, array + 47, array + 48, array + 49);
    return;
  case 51:
    minmax51(array, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29, array + 30, array + 31, array + 32, array + 33, array + 34, array + 35, array + 36, array + 37, array + 38, array + 39, array + 40, array + 41, array + 42, array + 43, array + 44, array + 45, array + 46, array + 47, array + 48, array + 49, array + 50);
    return;
  case 52:
    minmax52(array, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29, array + 30, array + 31, array + 32, array + 33, array + 34, array + 35, array + 36, array + 37, array + 38, array + 39, array + 40, array + 41, array + 42, array + 43, array + 44, array + 45, array + 46, array + 47, array + 48, array + 49, array + 50, array + 51);
    return;
  case 53:
    minmax53(array, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29, array + 30, array + 31, array + 32, array + 33, array + 34, array + 35, array + 36, array + 37, array + 38, array + 39, array + 40, array + 41, array + 42, array + 43, array + 44, array + 45, array + 46, array + 47, array + 48, array + 49, array + 50, array + 51, array + 52);
    return;
  case 54:
    minmax54(array, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29, array + 30, array + 31, array + 32, array + 33, array + 34, array + 35, array + 36, array + 37, array + 38, array + 39, array + 40, array + 41, array + 42, array + 43, array + 44, array + 45, array + 46, array + 47, array + 48, array + 49, array + 50, array + 51, array + 52, array + 53);
    return;
  case 55:
    minmax55(array, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29, array + 30, array + 31, array + 32, array + 33, array + 34, array + 35, array + 36, array + 37, array + 38, array + 39, array + 40, array + 41, array + 42, array + 43, array + 44, array + 45, array + 46, array + 47, array + 48, array + 49, array + 50, array + 51, array + 52, array + 53, array + 54);
    return;
  case 56:
    minmax56(array, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29, array + 30, array + 31, array + 32, array + 33, array + 34, array + 35, array + 36, array + 37, array + 38, array + 39, array + 40, array + 41, array + 42, array + 43, array + 44, array + 45, array + 46, array + 47, array + 48, array + 49, array + 50, array + 51, array + 52, array + 53, array + 54, array + 55);
    return;
  case 57:
    minmax57(array, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29, array + 30, array + 31, array + 32, array + 33, array + 34, array + 35, array + 36, array + 37, array + 38, array + 39, array + 40, array + 41, array + 42, array + 43, array + 44, array + 45, array + 46, array + 47, array + 48, array + 49, array + 50, array + 51, array + 52, array + 53, array + 54, array + 55, array + 56);
    return;
  case 58:
    minmax58(array, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29, array + 30, array + 31, array + 32, array + 33, array + 34, array + 35, array + 36, array + 37, array + 38, array + 39, array + 40, array + 41, array + 42, array + 43, array + 44, array + 45, array + 46, array + 47, array + 48, array + 49, array + 50, array + 51, array + 52, array + 53, array + 54, array + 55, array + 56, array + 57);
    return;
  case 59:
    minmax59(array, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29, array + 30, array + 31, array + 32, array + 33, array + 34, array + 35, array + 36, array + 37, array + 38, array + 39, array + 40, array + 41, array + 42, array + 43, array + 44, array + 45, array + 46, array + 47, array + 48, array + 49, array + 50, array + 51, array + 52, array + 53, array + 54, array + 55, array + 56, array + 57, array + 58);
    return;
  case 60:
    minmax60(array, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29, array + 30, array + 31, array + 32, array + 33, array + 34, array + 35, array + 36, array + 37, array + 38, array + 39, array + 40, array + 41, array + 42, array + 43, array + 44, array + 45, array + 46, array + 47, array + 48, array + 49, array + 50, array + 51, array + 52, array + 53, array + 54, array + 55, array + 56, array + 57, array + 58, array + 59);
    return;
  case 61:
    minmax61(array, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29, array + 30, array + 31, array + 32, array + 33, array + 34, array + 35, array + 36, array + 37, array + 38, array + 39, array + 40, array + 41, array + 42, array + 43, array + 44, array + 45, array + 46, array + 47, array + 48, array + 49, array + 50, array + 51, array + 52, array + 53, array + 54, array + 55, array + 56, array + 57, array + 58, array + 59, array + 60);
    return;
  case 62:
    minmax62(array, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29, array + 30, array + 31, array + 32, array + 33, array + 34, array + 35, array + 36, array + 37, array + 38, array + 39, array + 40, array + 41, array + 42, array + 43, array + 44, array + 45, array + 46, array + 47, array + 48, array + 49, array + 50, array + 51, array + 52, array + 53, array + 54, array + 55, array + 56, array + 57, array + 58, array + 59, array + 60, array + 61);
    return;
  case 63:
    minmax63(array, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29, array + 30, array + 31, array + 32, array + 33, array + 34, array + 35, array + 36, array + 37, array + 38, array + 39, array + 40, array + 41, array + 42, array + 43, array + 44, array + 45, array + 46, array + 47, array + 48, array + 49, array + 50, array + 51, array + 52, array + 53, array + 54, array + 55, array + 56, array + 57, array + 58, array + 59, array + 60, array + 61, array + 62);
    return;
  case 64:
    minmax64(array, array + 1, array + 2, array + 3, array + 4, array + 5, array + 6, array + 7, array + 8, array + 9, array + 10, array + 11, array + 12, array + 13, array + 14, array + 15, array + 16, array + 17, array + 18, array + 19, array + 20, array + 21, array + 22, array + 23, array + 24, array + 25, array + 26, array + 27, array + 28, array + 29, array + 30, array + 31, array + 32, array + 33, array + 34, array + 35, array + 36, array + 37, array + 38, array + 39, array + 40, array + 41, array + 42, array + 43, array + 44, array + 45, array + 46, array + 47, array + 48, array + 49, array + 50, array + 51, array + 52, array + 53, array + 54, array + 55, array + 56, array + 57, array + 58, array + 59, array + 60, array + 61, array + 62, array + 63);
    return;
  }
}
} // unnamed namespace

#endif /* MINMAX_64_CUH */