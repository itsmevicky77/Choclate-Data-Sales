{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e4412ad5",
   "metadata": {},
   "outputs": [
    {
     "ename": "",
     "evalue": "",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31mRunning cells with 'base (Python 3.11.13)' requires the ipykernel package.\n",
      "\u001b[1;31mInstall 'ipykernel' into the Python environment. \n",
      "\u001b[1;31mCommand: 'conda install -n base ipykernel --update-deps --force-reinstall'"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "# Display all columns\n",
    "pd.set_option(\"display.max_columns\", None)\n",
    "\n",
    "# Display numbers with 2 decimal places\n",
    "pd.set_option(\"display.float_format\", \"{:.2f}\".format)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "base",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.11.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
