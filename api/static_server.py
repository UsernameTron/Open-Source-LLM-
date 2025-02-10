from aiohttp import web
import os

async def handle_static(request):
    return web.FileResponse('./static/index.html')

app = web.Application()
app.router.add_get('/', handle_static)
app.router.add_static('/static/', path='./static')

if __name__ == '__main__':
    web.run_app(app, host='localhost', port=8080)
