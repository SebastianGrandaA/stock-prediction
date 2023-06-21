# stock-prediction

This project aims to identify the most important macroeconomic features and firm characteristics for predicting stock returns.
The model approach is based on the paper [Deep learning in Asset Pricing](https://arxiv.org/pdf/1904.00745.pdf) by Luyang Chen, Markus Pelgert and Jason Zhu.

## Usage

To start the app, execute the following:

```zsh
make up
```

Configure the model parameters using the `input/settings.json` file or directly editing the default parameters at `Constants.jl`.
Considering the default configurations, the execution will proceed as follows:

```zsh
[ Info: Epoch 5 | (fixed_discriminator_loss = 0.012335597590330129)
[ Info: Epoch 100 | (discriminator_loss = 0.02887351531621765, generator_loss = -0.012335597590330129, fixed_discriminator_loss = 0.012335597590330129) | Elapsed time 1.32
[ Info: Finish training on validation | Sharpe Ratio: 0.121
┌ Info: Finish training on all instances:
│  Row │ ID          predictions                        sharpe_ratio
│      │ String      Array                              Float64
│ ─────┼─────────────────────────────────────────────────────────────
│    1 │ train       [-0.00170221, -0.205378, 0.11416…         0.015
│    2 │ test        [0.00859562, -0.00136039, -0.008…        -0.059
└    3 │ validation  [0.0900033, -0.00604439, -0.0005…         0.121
899.918699 seconds (856.80 M allocations: 1009.035 GiB, 21.61% gc time, 1.49% compilation time)
```

## Pending work

* Hyperparameter optimization

## Contact

Contact `sebastian.grandaa@icloud.com` for more information.
