
	
		################################################
		########   MUSIC GENERATION with RNNs   ########
		###########   by Francesco Vidaich   ###########
		################################################


First, I'm sorry for the size of the folder, but in order to generate 
samples the preprocessed version of the dataset was required (you 
could run the 'pre-process' script, but it takes few minutes to build 
the preprocessed dataset).

---------------------------------------------------------------------

To generate MIDI samples, you will need the PyPianoRoll package, 
you can install it with:

	pip install pypianoroll

---------------------------------------------------------------------

To generate a sample that continues an initial random seed picked 
from the dataset (results in 'generated_tracks'), run the command:
	
	python trained_two_hands_model.py

---------------------------------------------------------------------

In order to convert the generated sample to the mp3 format, an external
tool is needed. The simplest way is to use an online converter, like
the one linked here (keep default settings):

	https://solmire.com/midi-to-mp3

Another good way of listening to the samples is importing them in an
online MIDI sequencer (lately this tool was down, but I'll link that anyway):

	https://onlinesequencer.net/import
	(or https://musicboxmaniacs.com/create/ , but it's worse)

--------------------------------------------------------------------

If you want to generate a sample with the original network (old 
version without the two hands), the results will probably be worse 
and you will have to build the corresponding preprocessed dataset 
with the 'numpy_pre-process_script.py' (this can take a while, 
probably it is not worth the effort).