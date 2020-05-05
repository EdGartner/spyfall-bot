"""Modified from gpt-2/src/sample.py"""
import tensorflow as tf

from gpt2.src import model

def top_k_logits(logits, k):
    if k == 0:
        # no truncation
        return logits

    def _top_k():
        values, _ = tf.nn.top_k(logits, k=k)
        min_values = values[:, -1, tf.newaxis]
        return tf.where(
            logits < min_values,
            tf.ones_like(logits, dtype=logits.dtype) * -1e10,
            logits,
        )
    return tf.cond(
       tf.equal(k, 0),
       lambda: logits,
       lambda: _top_k(),
    )


def top_p_logits(logits, p):
    """Nucleus sampling"""
    batch, _ = logits.shape.as_list()
    sorted_logits = tf.contrib.framework.sort(logits, direction='DESCENDING', axis=-1)
    cumulative_probs = tf.cumsum(tf.nn.softmax(sorted_logits, axis=-1), axis=-1)
    indices = tf.stack([
        tf.range(0, batch),
        # number of indices to include
        tf.maximum(tf.reduce_sum(tf.cast(cumulative_probs <= p, tf.int32), axis=-1) - 1, 0),
    ], axis=-1)
    min_values = tf.gather_nd(sorted_logits, indices)
    return tf.where(
        logits < min_values,
        tf.ones_like(logits) * -1e10,
        logits,
    )


def sample_sequence(*, hparams, length, weights, contexts, terminals=[], batch_size=None, temperature=1, top_k=0, top_p=1):

    N = len(contexts)
    def step(hparams, tokens, past=None):
        lm_output = model.model(hparams=hparams, X=tokens, past=past, reuse=tf.AUTO_REUSE)

        logits = lm_output['logits'][:, :, :hparams.n_vocab]
        present = lm_output['present']
        present.set_shape(model.past_shape(hparams=hparams, batch_size=batch_size))
        return logits, present
        
    with tf.name_scope('sample_sequence'):
        def body(*loop_vars):
            pasts = loop_vars[:N]
            prev = loop_vars[N]
            output = loop_vars[N + 1]

            all_logits = []
            presents = []
            for past in pasts:
                l, p = step(hparams, prev, past)
                all_logits.append(l[:, -1, :] / tf.cast(temperature, tf.float32))
                presents.append(p)

            # samples from distribution that is linear combination of distribution
            # from each state
            #probs = tf.nn.softmax(tf.stack(all_logits, axis=-1), axis=-1)
            #logits = tf.math.log(tf.tensordot(probs, weights, axes=1))
            logits = tf.tensordot(tf.stack(all_logits, axis=-1), weights, axes=1)
            logits = top_k_logits(logits, k=top_k)
            logits = top_p_logits(logits, p=top_p)
            samples = tf.multinomial(logits, num_samples=1, output_dtype=tf.int32)
            for i in range(N):
                presents[i] = tf.concat([pasts[i], presents[i]], axis=-2)
            return presents + [samples] + [tf.concat([output, samples], axis=1)]
        
        all_logits = []
        presents = []
        for cntx in contexts:
            l, p = step(hparams, cntx, None)
            all_logits.append(l[:, -1, :])
            presents.append(p)
        logits = tf.tensordot(tf.stack(all_logits, axis=-1), weights, axes=1)
        logits /= tf.cast(temperature, tf.float32) * tf.norm(weights, ord=1)
        logits = top_k_logits(logits, k=top_k)
        logits = top_p_logits(logits, p=top_p)
        samples = tf.multinomial(logits, num_samples=1, output_dtype=tf.int32)
        loop_vars = presents + [samples] + [samples]
        
        def cond(*args):
            # its stupid that there's not a trivial way of doing this     
            prev_flat = tf.reshape(args[N], [-1])
            vals = [tf.equal(prev_flat, sym) for sym in terminals]
            return tf.math.logical_not(tf.cast(tf.count_nonzero(tf.stack(vals)), tf.bool))
                
        past_shape = model.past_shape(hparams=hparams, batch_size=batch_size)
        shape_invariants = [tf.TensorShape(past_shape)] * N + [tf.TensorShape([batch_size, None])] * 2
        final_vars = tf.while_loop(
            cond=cond, body=body,
            maximum_iterations=length - 1,
            loop_vars=loop_vars,
            shape_invariants=shape_invariants,
            back_prop=False,
        )

        return final_vars[-1]


def sequence_lprob(*, hparams, context, sequence, batch_size=None):

    def step(hparams, tokens, next_word, past=None):
        lm_output = model.model(hparams=hparams, X=tokens, past=past, reuse=tf.AUTO_REUSE)

        logit = lm_output['logits'][:, -1, next_word]
        logit = tf.expand_dims(logit, 1)
        presents = lm_output['present']
        presents.set_shape(model.past_shape(hparams=hparams, batch_size=batch_size))
        return logit, presents

    with tf.name_scope('sequence_perlex'):
        context = tf.convert_to_tensor(context)
        sequence = tf.convert_to_tensor(sequence)

        def body(index, past, logits):
            logit, presents = step(hparams, sequence[:, index-1:index], sequence[0, index], past=past)
            return [
                index + tf.constant(1),
                tf.concat([past, presents], axis=-2),
                tf.concat([logits, logit], axis=-1)
            ]

        index = tf.constant(1)
        logits, past = step(hparams, context, sequence[0, index], None)

        def cond(*args):
            return True

        _, _, logits = tf.while_loop(
            cond=cond, body=body,
            maximum_iterations=tf.shape(sequence)[1] - 1,
            loop_vars=[
                index,
                past,
                logits
            ],
            shape_invariants=[
                tf.TensorShape([]),
                tf.TensorShape(model.past_shape(hparams=hparams, batch_size=batch_size)),
                tf.TensorShape([batch_size, None]),
            ],
            back_prop=False,
        )

        return tf.math.divide(tf.reduce_sum(logits), tf.cast(tf.size(logits), tf.float32))