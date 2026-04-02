# -*- coding: latin-1 -*-
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def encode_weights_to_unicode_string(weights, offset=12.0, divider=2048.0):# 12 2048 / 24  1024
	"""
	Encode un tableau numpy float en chane unicode compresse.
	"""
	print("min max:", weights.min(), weights.max())
	weights = np.clip(weights, -offset, offset)
	s = np.round(divider * (weights + offset)).astype(np.uint16)
	bytes_be = bytearray()
	s_flat = s.flatten()

	for val in s_flat:
		bytes_be.append((val >> 8) & 0xFF)
		bytes_be.append(val & 0xFF)
	unicode_str = bytes_be.decode('utf-16-be')
	return unicode_str


def decode_unicode_string_to_weights10(unicode_str, offset=12.0, divider=2048.0, shape=None):
	"""
	Dcode la chane unicode compresse en poids float numpy.
	"""
	bytes_be = unicode_str.encode('utf-16-be')
	arr_uint16 = np.frombuffer(bytes_be, dtype=np.uint16)
	weights = arr_uint16.astype(np.float32) / divider - offset
	if shape is not None:
		weights = weights.reshape(shape)

	# Affichage debug
	#print(f"Poids dcompresss (shape={weights.shape}):")
	#print(weights)
	return weights

def decode_unicode_string_to_weights(unicode_str, offset=12.0, divider=2048.0, shape=None):
    # �tape 1 : reconstruire la cha�ne binaire 'weights_bytes' comme en C++ wstring -> string
    weights_bytes = bytearray()
    for c in unicode_str:
        val = ord(c)
        weights_bytes.append((val >> 8) & 0xFF)  # octet haut
        weights_bytes.append(val & 0xFF)         # octet bas

    # �tape 2 : lire les poids 2 octets par 2 octets, big-endian
    size = len(weights_bytes) // 2
    output = []
    for i in range(size):
        s1 = weights_bytes[2*i]
        s2 = weights_bytes[2*i + 1]
        s = (s1 << 8) + s2
        val = (s / divider) - offset
        output.append(val)

    # �tape 3 : si shape pr�cis�, reshape en numpy array
    if shape is not None:
        import numpy as np
        output = np.array(output, dtype=np.float32).reshape(shape)
    else:
        output = list(output)

    return output

def export_weights_decompressed_to_txt(decompressed_weights_dict, output_txt='weights_decompressed.txt'):
	"""
	Enregistre les poids dcompresss dans un fichier texte lisible.
	"""
	with open(output_txt, 'w', encoding='utf-8') as f:
		for name, weights in decompressed_weights_dict.items():
			f.write(f"### {name} (shape={weights.shape}) ###\n")
			weights_str = np.array2string(weights, precision=6, floatmode='fixed', max_line_width=120)
			f.write(weights_str + '\n\n')

	print(f"[o] Poids dcompresss enregistrs dans {output_txt}")


def export_torch_weights_to_unicode_python_file(model_class, checkpoint_path,
											   output_py='weights_unicode.py',
											   output_txt='weights_decompressed.txt'):
	"""
	Exporte les poids compresss dans un fichier Python
	et enregistre les poids dcompresss dans un fichier texte lisible.
	"""
	model = model_class()
	#checkpoint = torch.load(checkpoint_path, map_location='cpu')
	#model.load_state_dict(checkpoint['model_state_dict'])

	model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
	model.eval()

	decompressed_weights_dict = {}

	with open(output_py, 'w', encoding='utf-8') as f:
		f.write("# Weights encoded as UTF-16BE unicode strings with offset=12, divider=2048\n\n")

		for name, param in model.named_parameters():
			print(name, param.shape)
			np_array = param.detach().cpu().numpy().astype(np.float32)
			
			encoded_str = encode_weights_to_unicode_string(np_array)
			safe_name = name.replace('.', '_')
			shape_str = str(np_array.shape)
			#print(f"{safe_name}", np_array)
			#f.write(f"{safe_name}_shape = {shape_str}\n")
			f.write(f"wstring {safe_name}_pol = L\"{encoded_str}\";\n\n")

			decompressed = decode_unicode_string_to_weights(encoded_str, shape=np_array.shape)
			decompressed_weights_dict[safe_name] = decompressed

	export_weights_decompressed_to_txt(decompressed_weights_dict, output_txt)

	print(f"[o] Poids compresss exports dans {output_py}")


# Exemple dutilisation
if __name__ == "__main__":
	
	class PolicyNet(nn.Module):
		def __init__(self, input_size=24):
			super().__init__()
			self.net = nn.Sequential(
				nn.Linear(input_size, 64),
				nn.ReLU(),
				nn.Linear(64, 32),
				nn.ReLU(),
				nn.Linear(32, 4)  # 4 directions
			)

		def forward(self, x):
			return torch.softmax(self.net(x), dim=-1)


	# Remplace 'checkpoint6uslim.pth' par le chemin de ton checkpoint
	export_torch_weights_to_unicode_python_file(
		model_class=PolicyNet,
		checkpoint_path='policy.pth',
		output_py='weights_unicode128PUCT.py',
		output_txt='weights_decompressed.txt'
	)
