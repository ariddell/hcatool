import os

import numpy as np
import pandas as pd


def load_counts(datastem, fitstem):
    """Returns topic assignment counts

    Returns topic-document assignment counts and topic-word assignment counts
    for a single iteration `fitstem`.

    Parameters
    ----------
    datastem : string
    fitstem : string

    Output
    ------
    document-topic, document-word : (DataFrame, DataFrame)
    """
    vocab_fn = '{}.tokens'.format(datastem)
    docnames_fn = '{}.documents'.format(datastem)
    vocab = tuple(open(vocab_fn).read().split())

    with open('{}.ndt'.format(fitstem)) as f:
        num_docs = int(f.readline().strip())
        num_topics = int(f.readline().strip())
        f.readline()  # discard third line
        ndt = np.zeros((num_docs, num_topics))
        for line in f:
            d, t, cnt = (int(elem) for elem in line.split())
            ndt[d, t] = cnt
    assert (ndt >= 0).all()
    try:
        docnames = tuple(os.path.splitext(os.path.basename(n))[0] for n in open(docnames_fn).read().split())
    except FileNotFoundError:
        docnames = tuple(range(num_docs))
    np.testing.assert_equal(num_docs, len(docnames))
    np.testing.assert_equal(len(docnames), len(ndt))
    ndt_df = pd.DataFrame(ndt, index=docnames)
    assert ndt_df.index.is_unique

    with open('{}.nwt'.format(fitstem)) as f:
        vocab_size = int(f.readline().strip())
        num_topics = int(f.readline().strip())
        f.readline()  # discard third line
        np.testing.assert_equal(vocab_size, len(vocab))
        ntw = np.zeros((num_topics, vocab_size))
        for line in f:
            w, t, cnt = (int(elem) for elem in line.split())
            ntw[t, w] = cnt
    assert (ntw >= 0).all()
    ntw_df = pd.DataFrame(ntw, columns=vocab)
    return (ndt_df, ntw_df)


def load(datastem, fitstem):
    """Returns posterior parameter estimates

    Returns posterior estimates for topic-document and topic-word distributions
    counts for a single iteration `fitstem`. Uses hyperparameter estimates
    from HCA (stored in the `fitstem.par` file).

    Parameters
    ----------
    datastem : string
    fitstem : string

    Output
    ------
    document-topic, document-word : (DataFrame, DataFrame)
    """
    ndt_df, ntw_df = load_counts(datastem, fitstem)
    with open('{}.par'.format(fitstem)) as f:
        params = [line.split(' = ') for line in f if not line.startswith('#')]
    num_doc, num_topic = ndt_df.shape
    _, num_vocab = ntw_df.shape
    for param, value in params:
        if param == 'alphatot':
            alphatot = float(value)
        elif param == 'betatot':
            betatot = float(value)
        elif param == 'D':
            np.testing.assert_equal(int(value), num_doc)
        elif param == 'T':
            np.testing.assert_equal(int(value), num_topic)
        elif param == 'W':
            np.testing.assert_equal(int(value), num_vocab)
    assert alphatot > 0
    assert betatot > 0
    theta_df = ndt_df + (alphatot / num_topic)
    theta_df = theta_df.div(theta_df.sum(axis=1), axis=0)
    phi_df = ntw_df + (betatot / num_vocab)
    phi_df = phi_df.div(phi_df.sum(axis=1), axis=0)
    return theta_df, phi_df
