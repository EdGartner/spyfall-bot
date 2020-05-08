import tensorflow as tf
import numpy as np
import os
import json

import spyfall_sample
from gpt2.src import encoder
from gpt2.src import model

class Bot:

    def __init__(self, prompts, prompt_index, players, position, 
                model_name='355M', models_dir='gpt2/models'):
        """
        : prompts: list of all the potential prompts informing non-spy players 
        of the secret location
        : prompt_index: index of the prompt for this game in prompts list
        : players: total number of players in the game
        : position: number of this bot in the order of player
        : model_name: which dir in model_dir
        : model_dir: path to language model used
        """

        if prompt_index == None:
            self.spy = True
        else:
            self.spy = False
            self.index = prompt_index

        self.players = players
        self.position = position
        self.hide_info = hide_info

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
            self.model_generate = spyfall_sample.sample_sequence(
                hparams=self.hparams, 
                length=50,
                weights=self.gen_weights,
                contexts=self.gen_contexts,
                terminals=self.enc.encode('.') + self.enc.encode('!') + self.enc.encode('?'),
                batch_size=self.batch_size,
                temperature=0.5, top_k=0, top_p=0.9
            )
            self.p_context = tf.placeholder(tf.int32, [self.batch_size, None])
            self.p_sequence = tf.placeholder(tf.int32, [self.batch_size, None])
            self.model_probability = spyfall_sample.sequence_lprob(
                hparams=self.hparams,
                context=self.p_context,
                sequence=self.p_sequence,
                batch_size=self.batch_size
            )
            saver = tf.train.Saver()
            ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
            saver.restore(self.sess, ckpt)
        
    def close(self):
        self.sess.close()
        
    def generate(self, transcript, hide_info=0.25):
        """ Returns a sentence based on prompt (if known) and transcript. 0 <= hide_info <= 1 decribes
        how much to weigh the other prompts relative to the real prompt. """
        if self.spy:
            weights = np.ones_like(self.prompts)
        else:
            weights = self.hide_info * np.ones_like(self.prompts)
            weights[self.index] = 1
        
        text = ' '.join(transcript)
        contexts = [[prmt + self.enc.encode(text) for _ in range(self.batch_size)] for prmt in self.prompts]
        sequence = self.sess.run(self.model_generate, feed_dict={
            **{ k: v for k, v in zip(self.gen_contexts, contexts) },
            self.gen_weights: weights
        })
        sequence = [token for batch in sequence for token in batch]
        return self.enc.decode(sequence)

    def accuse(self, transcript):
        """ Returns position of the player that the bot believes is most likely 
            the spy """

        off_lprobs = [[] for _ in range(self.players)]
        on_lprobs = [[] for _ in range(self.players)]
        for num, sentence in enumerate(transcript):
            context = self.enc.encode(' '.join(transcript[:num]))
            sequence = self.enc.encode(sentence)
            for i, prmt in enumerate(self.prompts):
                p = self.sess.run(self.model_probability, feed_dict={
                    self.p_context: [prmt + context for _ in range(self.batch_size)],
                    self.p_sequence: [sequence for _ in range(self.batch_size)]
                })
                if i == self.index:
                    on_lprobs[num % self.players].append(p)
                else:
                    off_lprobs[num % self.players].append(p)
        scale = np.min([p for ls in off_lprobs for p in ls]) / 2 # for numerical stability
        off_probs = [np.mean(np.exp(np.subtract(ls, scale, dtype=np.float64))) for ls in off_lprobs]
        on_probs = [np.mean(np.exp(np.subtract(ls, scale, dtype=np.float64))) for ls in on_lprobs]
        self.saved = [a / b for a, b in zip(on_probs, off_probs)]
        self.saved.pop(self.position)
        return np.argmin(self.saved)

    def guess(self, transcript, confidence=0.75):
        """ Returns the most likely location from the bot's perspective with 
        a threshhold confidence, otherwise False """

        text = ' '.join(transcript)
        sequence = self.enc.encode(text)
        lprobs = []
        for prmt in self.prompts:
            lprobs.append(self.sess.run(self.model_probability, feed_dict={
                self.p_context: [prmt for _ in range(self.batch_size)],
                self.p_sequence: [sequence for _ in range(self.batch_size)]
            }))
        self.saved = lprobs
        scale = np.min(lprobs) / 2 # for numerical stability
        probs = np.exp(np.subtract(lprobs, scale, dtype=np.float64))
        probs /= np.linalg.norm(probs, 1)
        if np.max(probs) > confidence:
            return np.argmax(probs)
        return False

