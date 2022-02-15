"""
 * Copyright (C) 2016  Music Technology Group - Universitat Pompeu Fabra
 *
 * This file is part of jingjuPhoneticSegmentation
 *
 * pypYIN is free software: you can redistribute it and/or modify it under
 * the terms of the GNU Affero General Public License as published by the Free
 * Software Foundation (FSF), either version 3 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the Affero GNU General Public License
 * version 3 along with this program.  If not, see http://www.gnu.org/licenses/
 *
 * If you have any problem about this python version code, please contact: Rong Gong
 * rong.gong@upf.edu
 *
 *
 * If you want to refer this code, please use this article:
 *
"""

import warnings

try:
    from essentia.standard import Spectrum, Windowing, ERBBands, FrameGenerator
except ImportError:
    warnings.warn("essentia is not available.")
    ESSENTIA_AVAILABLE = False
else:
    ESSENTIA_AVAILABLE = True

import numpy as np
import scipy.signal


def mrcg(y, sample_rate=16000, frame1_length=400, frame2_length=3200, hop_length=160, num_bands=64):

    if not ESSENTIA_AVAILABLE:
        raise ImportError("Please import essentia to use mrcg")

    window_type = "hann"

    cochleagram1 = cochleagram(
        y,
        window_type=window_type,
        hop_length=hop_length,
        frame_length=frame1_length,
        sample_rate=sample_rate,
        num_bands=num_bands,
    )
    cochleagram2 = cochleagram(
        y,
        window_type=window_type,
        hop_length=hop_length,
        frame_length=frame2_length,
        sample_rate=sample_rate,
        num_bands=num_bands,
    )
    cochleagram3 = smooth(cochleagram1, v_span=5, h_span=5)
    cochleagram4 = smooth(cochleagram1, v_span=11, h_span=11)

    mrcg = np.hstack((cochleagram1, cochleagram2, cochleagram3, cochleagram4))

    return mrcg


def cochleagram(x, window_type, hop_length, frame_length, sample_rate, num_bands=64):
    n = 2 * frame_length  # padding 1 time frame_size
    spectrum = Spectrum(size=n)
    window = Windowing(type=window_type, zeroPadding=n - frame_length)
    high_frequency_bound = sample_rate / 2 if sample_rate / 2 < 11000 else 11000
    erb_bands = ERBBands(
        sampleRate=sample_rate,
        highFrequencyBound=high_frequency_bound,
        numberBands=num_bands,
        inputSize=frame_length + 1,
    )
    epsilon = np.finfo(np.float).eps

    cochleas = []
    for frame in FrameGenerator(x, frameSize=frame_length, hopSize=hop_length):
        frame = window(frame)
        spectrum_frame = spectrum(frame)
        erb_frame = np.log10(erb_bands(spectrum_frame) + epsilon)
        cochleas.append(erb_frame)
    return np.array(cochleas)


def smooth(m, v_span, h_span):
    # This function produces a smoothed version of cochleagram
    fil_size = (2 * v_span + 1) * (2 * h_span + 1)
    meanfil = np.ones([1 + 2 * h_span, 1 + 2 * h_span], dtype=np.single)
    meanfil = np.divide(meanfil, fil_size)

    out = scipy.signal.convolve2d(m, meanfil, boundary="fill", fillvalue=0, mode="same")
    return out
