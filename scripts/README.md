After running the masksk creation, check the log of the unrecognized faces, then re move the un recognized faces from the **.lst** file for trainning and the **pairs.txt** file for testing.

* Verificaton on agedb30 is not recognized 251_8963
* change run file to save the thresold!!
* git add keep to logs

# The wrong masks created on the verification files:


------------------ agedb30 ------------------
191/191_7341.jpg
382/382_14098.jpg
247/247_8796.jpg
153/153_5769.jpg
338/338_12241.jpg
427/427_15914.jpg
332/332_11996.jpg
76/76_2920.jpg
92/92_3474.jpg
148/148_5604.jpg
62/62_2458.jpg
153/153_5770.jpg

*** in PAIRS.TXT = 31 occurances ***
# (12/4000) - % of faults
# 5298 - total unique images in pairs file
# 31 - the number of rows(no two missings in the same row), that are containing the 12 fault images that were found manually.
# 6000 - number of pairs
((5298*(12/4000))*(31/12))/6000 ~ 0.7%


------------------ lfw ------------------
Doris_Schroeder_0002.jpg
Brad_Garrett_0002.jpg
Tatiana_Shchegoleva_0001.jpg
Nancy_Powell_0001.jpg
Ken_Watanabe_0001.jpg

*** in PAIRS.TXT = ? occurances ***
# (5/4000) - % of faults
# 7701 - total unique images in pairs file
# 9 - the number of rows(no two missings in the same row), that are containing the 5 fault images that were found manually.
# 6000 - number of pairs
((7701*(5/4000))*(9/5))/6000 ~ 0.3%
