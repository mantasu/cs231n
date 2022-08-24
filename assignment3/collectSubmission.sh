#!/bin/bash
set -euo pipefail

CODE=(
	"cs231n/rnn_layers.py"
	"cs231n/transformer_layers.py"
	"cs231n/classifiers/rnn.py"
	"cs231n/net_visualization_pytorch.py"
	"cs231n/gan_pytorch.py"
	"cs231n/simclr/contrastive_loss.py"
	"cs231n/simclr/data_utils.py"
	"cs231n/simclr/utils.py"
)
NOTEBOOKS=(
	"RNN_Captioning.ipynb"
	"Transformer_Captioning.ipynb"
	"Network_Visualization.ipynb"
	"Generative_Adversarial_Networks.ipynb"
	"Self_Supervised_Learning.ipynb"
	"LSTM_Captioning.ipynb"
)
PDFS=(
	"RNN_Captioning.ipynb"
  "Transformer_Captioning.ipynb"
	"Network_Visualization.ipynb"
	"Generative_Adversarial_Networks.ipynb"
)

FILES=( "${CODE[@]}" "${NOTEBOOKS[@]}" )
ZIP_FILENAME="a3_code_submission.zip"
PDF_FILENAME="a3_inline_submission.pdf"

for FILE in "${FILES[@]}"
do
	if [ ! -f ${FILE} ]; then
		echo -e "${C_R}Required file ${FILE} not found, Exiting.${C_E}"
		exit 0
	fi
done

echo -e "### Zipping file ###"
rm -f ${ZIP_FILENAME}
zip -q "${ZIP_FILENAME}" -r ${NOTEBOOKS[@]} $(find . -name "*.py") $(find . -name "*.pyx") -x "makepdf.py"

echo -e "### Creating PDFs ###"
python makepdf.py --notebooks "${PDFS[@]}" --pdf_filename "${PDF_FILENAME}"

echo -e "### Done! Please submit ${ZIP_FILENAME} and ${PDF_FILENAME} to Gradescope. ###"
