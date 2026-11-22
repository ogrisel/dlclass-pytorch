# Slides

Each lecture is a [remark.js](https://remarkjs.com/) slideshow: `slides.md`
holds the content and `index.html` loads it in the browser.

## Viewing the slides locally

The slides load `slides.md` with `fetch()`, which browsers block on `file://`
URLs. Serve this folder over HTTP instead. From a local checkout of the
repository:

```sh
cd slides
python3 -m http.server 8000
```

Then open one of these URLs in Firefox (or any other browser):

- Object detection: <http://localhost:8000/convnets_detection/>
- Semantic and instance segmentation: <http://localhost:8000/convnets_segmentation/>

To open them directly from a terminal:

```sh
# macOS
open -a Firefox http://localhost:8000/convnets_detection/

# Linux
firefox http://localhost:8000/convnets_detection/
```

Stop the server with `Ctrl+C`. Navigate the slides with the arrow keys, or
press `p` for presenter mode.

The page loads `remark-latest.min.js` from `remarkjs.com`, so the first load
needs an internet connection.
