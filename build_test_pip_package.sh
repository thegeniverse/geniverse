if [ $PYPI_TEST_PWD ]
then
    echo "USING PASSWORD FROM PYPI_TEST_PWD"
else
    echo "NOT USING PASSWORD FROM PYPI_TEST_PWD"
fi

python -m pip install --upgrade build;
python -m build;
python -m pip install --upgrade twine;
python -m twine upload --repository testpypi ./dist/* --skip-existing --user viccpoes --password $PYPI_TEST_PWD
