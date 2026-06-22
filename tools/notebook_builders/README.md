# Notebook 生成脚本

各变体的 `*.ipynb` 由这里的 `build_nbNN.py` 用 `nbformat` 生成，再用
`jupyter nbconvert --to notebook --execute --inplace` 执行以嵌入真实输出/图表。

要修改某个 notebook：改对应的 `build_nbNN.py`，然后：

```bash
python tools/notebook_builders/build_nb03.py        # 生成
jupyter nbconvert --to notebook --execute --inplace 03-gqa-mqa/gqa.ipynb   # 执行嵌入输出
```

约定：tutorial 叙事（提问→动手→发现）；有 kernel 的变体用 `IPython.display.Code`
显示完整 kernel 源码（语法高亮）+ markdown 逐段精读。
