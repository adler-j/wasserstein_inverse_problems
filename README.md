Learning to solve inverse problems using Wasserstein loss
=========================================================

This repository contains the code for the article "Learning to solve inverse problems using Wasserstein loss".

Contents
--------
The code contains the following

* Training using circle phantoms

Pre-trained networks
--------------------
The pre-trained networks are currently under finalization and will be released soon, in the meantime, training is just a few hours.

Dependencies
------------
The code is currently based on the latest version of [ODL](https://github.com/odlgroup/odl) and the utility library [adler](https://github.com/adler-j/adler). They can be most easily installed by running 

```bash
$ pip install https://github.com/odlgroup/odl/archive/master.zip
$ pip install https://github.com/adler-j/adler/archive/master.zip
```

The learning requires tensorflow, and the ray-transform needs [ASTRA](https://github.com/astra-toolbox/astra-toolbox) for computational feasibility

```bash
$ conda install -c astra-toolbox astra-toolbox
```

Authors
-------
[Jonas Adler](https://www.kth.se/profile/jonasadl), PhD student  
KTH, Royal Institute of Technology  
Elekta  
jonasadl@kth.se

[Axel Ringh](https://www.kth.se/profile/aringh), PhD student  
KTH, Royal Institute of Technology  
aringh@kth.se

[Ozan Öktem](https://www.kth.se/profile/ozan), Associate Professor  
KTH, Royal Institute of Technology  
ozan@kth.se

[Johan Karlsson](https://people.kth.se/~johan79/), Associate Professor  
KTH, Royal Institute of Technology  
johan.karlsson@math.kth.se

Funding
-------
Development is financially supported by the Swedish Foundation for Strategic Research as part of the project "Low complexity image reconstruction in medical imaging" and "3D reconstruction with simulated forward models".

Development has also been financed by [Elekta](https://www.elekta.com/).
