# Third-party material

The bundled public landmark data in `datasets/signn.json`, the prepared coordinate dataset, and model artifacts derived from that data originate from [AriAlavi/SigNN](https://github.com/AriAlavi/SigNN), revision `40bd651899e09a0099767457194062f77f774156`.

SigNN's upstream repository license is Apache License 2.0. A copy is preserved at `datasets/SIGNN-LICENSE.txt`. Provenance and limitations are recorded in `datasets/provenance.json`. Modifications here include deduplication, coordinate normalization, grouping unsupported labels, training a new classifier, and exporting browser inference weights and evaluation artifacts.

Upstream contributors listed in the SigNN README: Arian Alavi, Vahe Barseghyan, Rafael Trinidad, Gokul Deep, Kenny Yip, John Kirchner, Daniel Lohn, and Conor O'Brien.

MediaPipe and TensorFlow.js are loaded from their public distributions. Their licenses remain with their distributions. ASL University / Lifeprint learning references are linked to their original website. Interface icons and hand illustration are original SVG source within this project.

The added J/Z motion sequences and learned movement model derive from the SigNN Team public video dataset, version 1, published on Kaggle under CC0: Public Domain. Source: https://www.kaggle.com/datasets/signnteam/asl-sign-language-alphabet-videos-j-z . `datasets/motion-provenance.json` records exact file names and hashes. Raw clips are ignored by Git; retained coordinate sequences are extracted by MediaPipe and used to train a new temporal classifier.
