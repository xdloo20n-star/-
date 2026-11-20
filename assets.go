package sverka

import "embed"

//go:embed docs/extractor.schema.json
var ExtractorSchema []byte

//go:embed docs/examples/extractor-mock.json
var MockExtraction []byte

//go:embed web
var WebAssets embed.FS
