README 7x7x7x10_raw
-----------------
rData0*_230414_7x7x7x10_*

emg_reponce: Matrix containing the responce value (ex: EMG peak responce) in a window time of 40 ms after the stimulation offset.
 	dim 1 : number of repetition for a specific combination
 	dim 2 : emg channel (4)
 	dim 3 : specific combination --> comb i, corresponds to a combination of stim parameter of the matrix stim_combination
stim_combination: Matrix containing the unique combinations of the stimulation parameters:
		[PW (us), freq (Hz), stim_train_duration (ms), numb_pulses(count)]

NB: the mapping of the implant is the following: 

     1    10    26    17
     3    12    28    19
     5    14    30    21
     7    16    32    23
     2     9    25    18
     4    11    27    20
     6    13    29    22
     8    15    31    24