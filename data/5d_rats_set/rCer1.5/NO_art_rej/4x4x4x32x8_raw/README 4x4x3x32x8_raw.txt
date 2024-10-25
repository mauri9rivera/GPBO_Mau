README 4x4x3x32x8_raw
-----------------
rData0*_230414_4x4x3x32x8_*

emg_reponce: 	Matrix containing the response value (ex: EMG peak response) in a window time
 		of 40ms seconds after the stimulation offset.
 			dim 1 : number of repetitions (8)
 			dim 2 : emg channel number (4)
 			dim 3 : number of combinations (1536) --> specific combination comb i corresponds to a combination of stim parameter of the matrix stim_combination

stim_combination: Matrix containing the unique combinations of the stimulation parameters:
		[PW (us), freq (Hz), stim_train_duration (ms), count, channel, x_ch, y_ch]

NB: the mapping of the implant is the following: 

     1    10    26    17
     3    12    28    19
     5    14    30    21
     7    16    32    23
     2     9    25    18
     4    11    27    20
     6    13    29    22
     8    15    31    24