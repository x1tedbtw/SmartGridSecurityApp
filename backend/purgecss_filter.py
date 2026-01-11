from compressor.filters import CompilerFilter
from django.conf import settings
from pathlib import Path


COMPRESS_PURGECSS_BINARY = "purgecss"
COMPRESS_PURGECSS_ARGS = ""
COMPRESS_PURGECSS_APPS_INCLUDED = []

class PurgeCSSFilter(CompilerFilter):
    command = '{binary} --css {infile} -o {outfile} {args}'
    options = (
        ("binary",
         getattr(
             settings,
             "COMPRESS_PURGECSS_BINARY",
             COMPRESS_PURGECSS_BINARY)),
        ("args",
         getattr(
            settings,
            "COMPRESS_PURGECSS_ARGS",
            COMPRESS_PURGECSS_ARGS)),
             )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        template_files = " ".join(self.get_all_template_files())
        self.command += " --content {}".format(template_files)

    def get_all_template_files(self):
        files = []
        dir = Path(settings.BASE_DIR) / 'templates'
        for f in dir.glob('**/*.html'):
            files.append(str(f))
        return files