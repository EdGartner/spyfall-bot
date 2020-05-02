We have to be done with everything in a week, so I was hoping we could finish the coding this weekend. 

__Components__
* bot : takes the language model and transcript and acts as a player in the game (Liam can do this)
   * generator : returns a next sentence. needs to match the secret prompt without revealing too much information
   * detector : estimates the probability that the speaker knows the location
   * guesser : predicts probabilies of each location based on the transcript
* model : takes an arbitary sequence of characters and returns its probability (GPT-2 or anything we want to test)
* game : assigns a player the spy, keeps track of the transcript, allows players to nominate a spy, resolves voting on a spy and the game outcome
   * interface : some way for humans to play the game. could be just command line 
   * location prompts : list of all the secret prompts, one of which is selected at the start of the game

I have started working on the bot generator. I also downloaded the GPT-2 models and played around with it a little, but was having a hard time understanding how to use it.