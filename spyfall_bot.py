import tensorflow as tf
import numpy as np
import os
import json

import spyfall_sample
from gpt2.src import encoder
from gpt2.src import model

class Bot:

    def __init__(self, prompts, prompt_index, players, position, model_name='355M', models_dir='gpt2/models'):
        """
        : prompts: list of all the potential prompts informing non-spy players 
        of the secret location
        : index: index of the prompt for this game in prompts list
        : players: total number of players in the game
        : position: number of this bot in the order of player
        : model_dir: path to language model used
        """

        if prompt_index == None:
            self.spy = True
        else:
            self.spy = False
            self.index = prompt_index

        self.players = players
        self.position = position
        self.accuse_confidence = 0.6 + 0.4 / (self.players - 1)
        self.guess_confidence = 0.8 - 0.4 / (self.players - 1)
        self.hide_info = 0.2

        self.batch_size = 1
        models_dir = os.path.expanduser(os.path.expandvars(models_dir))
        self.enc = encoder.get_encoder(model_name, models_dir)
        self.prompts = [self.enc.encode(text) for text in prompts]
        self.hparams = model.default_hparams()
        with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
            self.hparams.override_from_dict(json.load(f))

        graph = tf.Graph()
        self.sess = tf.Session(graph=graph)
        with graph.as_default():
            self.gen_contexts = [tf.placeholder(tf.int32, [self.batch_size, None]) for _ in range(len(prompts))]
            self.gen_weights = tf.placeholder(tf.float32, [len(prompts)])
            self.generator = spyfall_sample.sample_sequence(
                hparams=self.hparams, 
                length=50,
                weights=self.gen_weights,
                contexts=self.gen_contexts,
                terminals=self.enc.encode('.') + self.enc.encode('!') + self.enc.encode('?'),
                batch_size=self.batch_size,
                temperature=0.5, top_k=0, top_p=0.9
            )
            self.plx_context = tf.placeholder(tf.int32, [self.batch_size, None])
            self.plx_sequence = tf.placeholder(tf.int32, [self.batch_size, None])
            self.perplexor = spyfall_sample.sequence_perplex(
                hparams=self.hparams,
                context=self.plx_context,
                sequence=self.plx_sequence,
                batch_size=self.batch_size
            )
            saver = tf.train.Saver()
            ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
            saver.restore(self.sess, ckpt)
        

    def close(self):
        self.sess.close()
        
    def _model(self, *args):
        # model.model
        pass

    def _predictions(self, transcript):
        text = ' '.join(transcript)
        return [self.model.predict(prompt + text) for prompt in self.prompts]
        
    def generate(self, transcript):
        if self.spy:
            weights = self._predictions(transcript)
        else:
            weights = self.hide_info * np.ones_like(self.prompts)
            weights[self.index] = 1
        
        text = ' '.join(transcript)
        contexts = [[prmt + self.enc.encode(text) for _ in range(self.batch_size)] for prmt in self.prompts]
        sequence = self.sess.run(self.generator, feed_dict={
            **{ k: v for k, v in zip(self.gen_contexts, contexts) },
            self.gen_weights: weights
        })
        sequence = [token for batch in sequence for token in batch]
        return self.enc.decode(sequence)

    def accuse(self, transcript, confidence = None):
        """ Returns whether the bot believes the last player who spoke is the 
        spy with a threshhold confidence """
        accused = len(transcript) // self.players
        np.true_divide(np.ones(self.players), self.players)

        if confidence == None:
            confidence = self.accuse_confidence

        pass #not done yet

    def guess(self, transcript, confidence = None):
        """ Returns the most likely location from the bot's perspective with 
        a threshhold confidence, otherwise False """
        if confidence == None:
            confidence = self.guess_confidence
        text = ' '.join(transcript)
        sequence = self.enc.encode(text)
        logits = []
        for prmt in self.prompts:
            logits.append(self.sess.run(self.perplexor, feed_dict={
                self.plx_context: [prmt for _ in range(self.batch_size)],
                self.plx_sequence: [sequence for _ in range(self.batch_size)]
            }))
        pv = np.power(2, np.multiply(-1/len(text), logits))
        probs = pv / np.linalg.norm(pv, 1)
        if np.max(probs) > confidence:
            return np.argmax(probs)
        return False

