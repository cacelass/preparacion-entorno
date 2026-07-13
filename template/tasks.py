from invoke import task


@task(help={
    'ip': 'IP to listen on, defaults to *',
    'port': 'Port to listen on, defaults to 8888',
})
def lab(ctx, ip='*', port=8888):
    """Launch Jupyter Lab"""
    ctx.run(f'jupyter lab --ip={ip} --port={port}', warn=True)


@task(help={
    'ip': 'IP to listen on, defaults to *',
    'port': 'Port to listen on, defaults to 8888',
})
def notebook(ctx, ip='*', port=8888):
    """Launch Jupyter Notebook"""
    ctx.run(f'jupyter notebook --ip={ip} --port={port}', warn=True)


@task
def data(ctx):
    """Run data pipeline step"""
    ctx.run('uv run python -m {{ project_slug }}.data.make_dataset')


@task
def features(ctx):
    """Run features pipeline step"""
    ctx.run('uv run python -m {{ project_slug }}.features.build_features')


@task
def train(ctx):
    """Run training pipeline step"""
    ctx.run('uv run python -m {{ project_slug }}.models.train_model')


@task
def predict(ctx):
    """Run prediction pipeline step"""
    ctx.run('uv run python -m {{ project_slug }}.models.predict_model')


@task(pre=[data, features, train, predict])
def pipeline(ctx):
    """Run full pipeline: data → features → train → predict"""
    ctx.run('echo "Pipeline completo."')


@task
def test(ctx):
    """Run tests"""
    ctx.run('uv run pytest tests/ -v')


@task
def lint(ctx):
    """Lint with ruff"""
    ctx.run('uv run ruff check {{ project_slug }}/ tests/')


@task
def format(ctx):
    """Format with ruff"""
    ctx.run('uv run ruff format {{ project_slug }}/ tests/')
