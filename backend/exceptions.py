import logging

logger = logging.getLogger(__name__)

class CustomExceptionMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        try:
            response = self.get_response(request)
        except ConnectionAbortedError:
            logger.warning("Connection aborted by the client.")
            return
        except Exception as e:
            logger.error("Unhandled exception: %s", e)
            raise
        return response