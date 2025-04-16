convert:
	uv tool run jupyter nbconvert --to python 01_binary.ipynb
	uv tool run jupyter nbconvert --to python 01_multiple.ipynb
	uv tool run jupyter nbconvert --to python apelid.ipynb
	uv tool run jupyter nbconvert --to python wgan.ipynb

	mv ./01_binary.py ./py/01_binary.py
	mv ./01_multiple.py ./py/01_multiple.py
	mv ./apelid.py ./py/apelid.py
	mv ./wgan.py ./py/wgan.py

py_convert:
	uv tool run p2j ./py/wgan.py -t wgan.ipynb -o
	uv tool run p2j ./py/apelid.py -t apelid.ipynb -o

