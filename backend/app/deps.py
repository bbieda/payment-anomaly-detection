from fastapi import Request


def get_db(request: Request):
    db = request.app.state.Session()
    try:
        yield db
    finally:
        db.close()


def get_store(request: Request):
    return request.app.state.store
