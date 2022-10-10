#!MC 1410
$!VarSet |MFBD| = 'C:\Program Files\Tecplot\Tecplot 360 EX 2017 R2\bin'
$!READDATASET  '"C:\Users\Home\Desktop\128128\softsign(1,2,3,4,5)@\23\23.txt" '
  READDATAOPTION = NEW
  RESETSTYLE = YES
  VARLOADMODE = BYNAME
  ASSIGNSTRANDIDS = YES
  VARNAMELIST = '"X" "Y" "Ureal" "Vreal" "Urec" "Vrec" "Uerror" "Verror"'
$!TWODAXIS AXISMODE = INDEPENDENT
$!TWODAXIS XDETAIL{RANGEMAX = 24}
$!TWODAXIS XDETAIL{RANGEMAX = 25}
$!TWODAXIS XDETAIL{RANGEMAX = 10}
$!TWODAXIS YDETAIL{RANGEMAX = 22}
$!PICK ADDATPOSITION
  X = 5.72459893048
  Y = 5.4050802139
  CONSIDERSTYLE = YES
$!REDRAWALL 
$!PICK ADDATPOSITION
  X = 1.45721925134
  Y = 6.20721925134
  CONSIDERSTYLE = YES
$!PICK ADDATPOSITION
  X = 10.013368984
  Y = 4.04679144385
  CONSIDERSTYLE = YES
$!FRAMECONTROL ACTIVATEBYNUMBER
  FRAME = 1
$!PICK COPY
$!PICK PASTE
$!PICK PASTE
$!EXTENDEDCOMMAND 
  COMMANDPROCESSORID = 'Multi Frame Manager'
  COMMAND = 'TILEFRAMESVERT'
$!REDRAWALL 
$!FRAMECONTROL ACTIVATEATPOSITION
  X = 1.41443850267
  Y = 0.763368983957
$!PICK ADDATPOSITION
  X = 1.41443850267
  Y = 0.763368983957
  CONSIDERSTYLE = YES
$!GLOBALRGB REDCHANNELVAR = 3
$!GLOBALRGB GREENCHANNELVAR = 3
$!GLOBALRGB BLUECHANNELVAR = 3
$!SETCONTOURVAR 
  VAR = 3
  CONTOURGROUP = 1
  LEVELINITMODE = RESETTONICE
$!SETCONTOURVAR 
  VAR = 4
  CONTOURGROUP = 2
  LEVELINITMODE = RESETTONICE
$!SETCONTOURVAR 
  VAR = 5
  CONTOURGROUP = 3
  LEVELINITMODE = RESETTONICE
$!SETCONTOURVAR 
  VAR = 6
  CONTOURGROUP = 4
  LEVELINITMODE = RESETTONICE
$!SETCONTOURVAR 
  VAR = 7
  CONTOURGROUP = 5
  LEVELINITMODE = RESETTONICE
$!SETCONTOURVAR 
  VAR = 8
  CONTOURGROUP = 6
  LEVELINITMODE = RESETTONICE
$!SETCONTOURVAR 
  VAR = 3
  CONTOURGROUP = 7
  LEVELINITMODE = RESETTONICE
$!SETCONTOURVAR 
  VAR = 3
  CONTOURGROUP = 8
  LEVELINITMODE = RESETTONICE
$!FIELDLAYERS SHOWCONTOUR = YES
$!FRAMECONTROL ACTIVATEATPOSITION
  X = 5.00802139037
  Y = 0.763368983957
$!PICK ADDATPOSITION
  X = 4.99732620321
  Y = 0.741978609626
  CONSIDERSTYLE = YES
$!GLOBALRGB REDCHANNELVAR = 3
$!GLOBALRGB GREENCHANNELVAR = 3
$!GLOBALRGB BLUECHANNELVAR = 3
$!SETCONTOURVAR 
  VAR = 3
  CONTOURGROUP = 1
  LEVELINITMODE = RESETTONICE
$!SETCONTOURVAR 
  VAR = 4
  CONTOURGROUP = 2
  LEVELINITMODE = RESETTONICE
$!SETCONTOURVAR 
  VAR = 5
  CONTOURGROUP = 3
  LEVELINITMODE = RESETTONICE
$!SETCONTOURVAR 
  VAR = 6
  CONTOURGROUP = 4
  LEVELINITMODE = RESETTONICE
$!SETCONTOURVAR 
  VAR = 7
  CONTOURGROUP = 5
  LEVELINITMODE = RESETTONICE
$!SETCONTOURVAR 
  VAR = 8
  CONTOURGROUP = 6
  LEVELINITMODE = RESETTONICE
$!SETCONTOURVAR 
  VAR = 3
  CONTOURGROUP = 7
  LEVELINITMODE = RESETTONICE
$!SETCONTOURVAR 
  VAR = 3
  CONTOURGROUP = 8
  LEVELINITMODE = RESETTONICE
$!FIELDLAYERS SHOWCONTOUR = YES
$!SETCONTOURVAR 
  VAR = 5
  CONTOURGROUP = 1
  LEVELINITMODE = RESETTONICE
$!FRAMECONTROL ACTIVATEATPOSITION
  X = 8.07754010695
  Y = 0.998663101604
$!PICK ADDATPOSITION
  X = 8.07754010695
  Y = 0.998663101604
  CONSIDERSTYLE = YES
$!GLOBALRGB REDCHANNELVAR = 3
$!GLOBALRGB GREENCHANNELVAR = 3
$!GLOBALRGB BLUECHANNELVAR = 3
$!SETCONTOURVAR 
  VAR = 3
  CONTOURGROUP = 1
  LEVELINITMODE = RESETTONICE
$!SETCONTOURVAR 
  VAR = 4
  CONTOURGROUP = 2
  LEVELINITMODE = RESETTONICE
$!SETCONTOURVAR 
  VAR = 5
  CONTOURGROUP = 3
  LEVELINITMODE = RESETTONICE
$!SETCONTOURVAR 
  VAR = 6
  CONTOURGROUP = 4
  LEVELINITMODE = RESETTONICE
$!SETCONTOURVAR 
  VAR = 7
  CONTOURGROUP = 5
  LEVELINITMODE = RESETTONICE
$!SETCONTOURVAR 
  VAR = 8
  CONTOURGROUP = 6
  LEVELINITMODE = RESETTONICE
$!SETCONTOURVAR 
  VAR = 3
  CONTOURGROUP = 7
  LEVELINITMODE = RESETTONICE
$!SETCONTOURVAR 
  VAR = 3
  CONTOURGROUP = 8
  LEVELINITMODE = RESETTONICE
$!FIELDLAYERS SHOWCONTOUR = YES
$!SETCONTOURVAR 
  VAR = 7
  CONTOURGROUP = 1
  LEVELINITMODE = RESETTONICE
$!FRAMECONTROL ACTIVATEATPOSITION
  X = 2.92245989305
  Y = 2.63502673797
$!PICK ADDATPOSITION
  X = 2.92245989305
  Y = 2.63502673797
  CONSIDERSTYLE = YES
$!GLOBALCONTOUR 1  LEGEND{ISVERTICAL = NO}
$!GLOBALCONTOUR 1  LEGEND{OVERLAYBARGRID = NO}
$!GLOBALCONTOUR 1  LABELS{AUTOLEVELSKIP = 2}
$!GLOBALCONTOUR 1  LABELS{AUTOLEVELSKIP = 3}
$!GLOBALCONTOUR 1  LABELS{AUTOLEVELSKIP = 4}
$!GLOBALCONTOUR 1  LEGEND{BOX{BOXTYPE = NONE}}
$!FRAMECONTROL ACTIVATEATPOSITION
  X = 6.32352941176
  Y = 3.0307486631
$!PICK ADDATPOSITION
  X = 6.32352941176
  Y = 3.0307486631
  CONSIDERSTYLE = YES
$!GLOBALCONTOUR 1  LEGEND{ISVERTICAL = NO}
$!GLOBALCONTOUR 1  LEGEND{OVERLAYBARGRID = NO}
$!GLOBALCONTOUR 1  LABELS{AUTOLEVELSKIP = 2}
$!GLOBALCONTOUR 1  LABELS{AUTOLEVELSKIP = 3}
$!GLOBALCONTOUR 1  LABELS{AUTOLEVELSKIP = 4}
$!GLOBALCONTOUR 1  LABELS{AUTOLEVELSKIP = 3}
$!GLOBALCONTOUR 1  LEGEND{BOX{BOXTYPE = NONE}}
$!FRAMECONTROL ACTIVATEATPOSITION
  X = 10.3128342246
  Y = 3.97192513369
$!PICK ADDATPOSITION
  X = 10.3128342246
  Y = 3.97192513369
  CONSIDERSTYLE = YES
$!GLOBALCONTOUR 1  LEGEND{OVERLAYBARGRID = NO}
$!GLOBALCONTOUR 1  LEGEND{OVERLAYBARGRID = YES}
$!GLOBALCONTOUR 1  LEGEND{OVERLAYBARGRID = NO}
$!GLOBALCONTOUR 1  LABELS{AUTOLEVELSKIP = 2}
$!GLOBALCONTOUR 1  LABELS{AUTOLEVELSKIP = 3}
$!GLOBALCONTOUR 1  LEGEND{BOX{BOXTYPE = NONE}}
$!FRAMECONTROL ACTIVATEATPOSITION
  X = 2.38770053476
  Y = 1.89705882353
$!PICK ADDATPOSITION
  X = 2.38770053476
  Y = 1.89705882353
  CONSIDERSTYLE = YES
$!PICK SHIFT
  X = -0.0641711229947
  Y = -1.37967914439
$!FRAMECONTROL ACTIVATEATPOSITION
  X = 5.25401069519
  Y = 2.12165775401
$!PICK ADDATPOSITION
  X = 5.25401069519
  Y = 2.12165775401
  CONSIDERSTYLE = YES
$!PICK SHIFT
  X = 0.0213903743315
  Y = -1.37967914439
$!FRAMECONTROL ACTIVATEATPOSITION
  X = 9.98128342246
  Y = 2.84893048128
$!PICK ADDATPOSITION
  X = 9.98128342246
  Y = 2.84893048128
  CONSIDERSTYLE = YES
$!PICK ADDATPOSITION
  X = 9.98128342246
  Y = 2.84893048128
  CONSIDERSTYLE = YES
$!GLOBALCONTOUR 1  LEGEND{ISVERTICAL = NO}
$!PICK ADDATPOSITION
  X = 10.8048128342
  Y = 5.38368983957
  CONSIDERSTYLE = YES
$!FRAMECONTROL ACTIVATEBYNUMBER
  FRAME = 3
$!PICK ADDATPOSITION
  X = 10.2486631016
  Y = 3.9077540107
  CONSIDERSTYLE = YES
$!REDRAWALL 
$!PICK ADDATPOSITION
  X = 9.20053475936
  Y = 2.08957219251
  CONSIDERSTYLE = YES
$!PICK SHIFT
  X = 0
  Y = -1.44385026738
$!EXPORTSETUP EXPORTREGION = ALLFRAMES
$!EXPORTSETUP IMAGEWIDTH = 988
$!EXPORTSETUP EXPORTFNAME = 'C:\Users\Home\Desktop\128128\softsign(1,2,3,4,5)@\23\U.png'
$!EXPORT 
  EXPORTREGION = ALLFRAMES
$!FRAMECONTROL ACTIVATEATPOSITION
  X = 0.954545454545
  Y = 1.97192513369
$!PICK ADDATPOSITION
  X = 0.954545454545
  Y = 1.97192513369
  CONSIDERSTYLE = YES
$!SETCONTOURVAR 
  VAR = 4
  CONTOURGROUP = 1
  LEVELINITMODE = RESETTONICE
$!FRAMECONTROL ACTIVATEATPOSITION
  X = 6.91176470588
  Y = 2.05748663102
$!PICK ADDATPOSITION
  X = 6.91176470588
  Y = 2.05748663102
  CONSIDERSTYLE = YES
$!SETCONTOURVAR 
  VAR = 4
  CONTOURGROUP = 1
  LEVELINITMODE = RESETTONICE
$!SETCONTOURVAR 
  VAR = 6
  CONTOURGROUP = 1
  LEVELINITMODE = RESETTONICE
$!FRAMECONTROL ACTIVATEATPOSITION
  X = 10.4732620321
  Y = 2.54946524064
$!PICK ADDATPOSITION
  X = 10.4732620321
  Y = 2.54946524064
  CONSIDERSTYLE = YES
$!SETCONTOURVAR 
  VAR = 8
  CONTOURGROUP = 1
  LEVELINITMODE = RESETTONICE
$!REDRAWALL 
$!EXPORTSETUP EXPORTFNAME = 'C:\Users\Home\Desktop\128128\softsign(1,2,3,4,5)@\23\V.png'
$!EXPORT 
  EXPORTREGION = ALLFRAMES
$!RemoveVar |MFBD|