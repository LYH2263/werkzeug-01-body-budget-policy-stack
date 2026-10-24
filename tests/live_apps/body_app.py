from werkzeug import body_budget
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.wrappers import Request
from werkzeug.wrappers import Response

# Tight per-request application-layer budget. The serving layer is
# deliberately not governed by this; it still drains the socket after a 413
# so the client receives a complete response instead of a reset connection.
BODY_LIMIT = 1024


@Request.application
def app(request):
    try:
        with body_budget(max_content_length=BODY_LIMIT):
            data = request.get_data()
    except RequestEntityTooLarge:
        return Response(status=413)

    return Response(f"read {len(data)} bytes", status=200)
