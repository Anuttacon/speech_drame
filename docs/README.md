# Speech-DRAME project page

This buildless page is served from `docs/index.html`. It has no external fonts, analytics, package dependencies, or data payloads.

The working preview is https://ftshijt.github.io/speech_drame/. It is published from a branch containing only the website files.

The code repository currently contains submodule entries without corresponding URLs in `.gitmodules`. GitHub's default Pages build initializes submodules recursively, so publishing directly from `main` → `/docs` fails during checkout. The project page does not need those submodules.

To publish under the organization after merging, a maintainer can create a site-only branch from this directory and push it:

```sh
git subtree split --prefix docs main -b codex/project-page-site
git push origin codex/project-page-site
```

A repository administrator can then select **Settings → Pages → Deploy from a branch → codex/project-page-site → /(root)**. GitHub assigns the final public URL. Repeat the subtree publication step for later page changes, using a new local branch name if needed, and update the existing remote site branch only with a normal fast-forward push. Alternatively, a custom Pages workflow can check out the source with `submodules: false` and deploy only `docs/`. Do not cite a Pages URL as live until deployment has been verified.

The page's scientific values and author list come from the camera-ready manuscript. Its release notices distinguish the experimental corpus from the authorized public subset and clearly state that the two Hugging Face pages do not yet contain benchmark data. Keep these notices current when the payload is released.
