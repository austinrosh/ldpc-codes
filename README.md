# EE 387 Final Project: Theory and Implementation of LDPC Codes

## Overview
This repository contains an implementation and simulation framework for LDPC (Low-Density Parity-Check) codes, supporting the final project deliverable for EE 387 at Stanford University.

The \simulation directory contains various scripts whcih explore channel coded simulations for 5G communications using the NVIDIA Sionna link-level modeler.

---

## Simulation Environment Setup

This project requires **Python 3.11** in order to be compatible with the latest Sionna release. Additionally, Sionna has package dependenices which need to be installed through the requirements file.

### 
Check if Python 3.11 is available:
```bash
python3.11 --version
python3.11 -m venv venv  
source venv/bin/activate  
pip install -r requirements.txt
```