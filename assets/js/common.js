// aHR0cHM6Ly9naXRodWIuY29tL2x1b3N0MjYvYWNhZGVtaWMtaG9tZXBhZ2U=
$(function () {
    lazyLoadOptions = {
        scrollDirection: 'vertical',
        effect: 'fadeIn',
        effectTime: 300,
        placeholder: "",
        onError: function(element) {
            console.log('[lazyload] Error loading ' + element.data('src'));
        },
        afterLoad: function(element) {
            if (element.is('img')) {
                // remove background-image style
                element.css('background-image', 'none');
                element.css('min-height', '0');
            } else if (element.is('div')) {
                // set the style to background-size: cover; 
                element.css('background-size', 'cover');
                element.css('background-position', 'center');
            }
        }
    }

    $('img.lazy, div.lazy:not(.always-load)').Lazy({visibleOnly: true, ...lazyLoadOptions});
    $('div.lazy.always-load').Lazy({visibleOnly: false, ...lazyLoadOptions});

    $('[data-toggle="tooltip"]').tooltip()

    var $grid = $('.grid').masonry({
        "percentPosition": true,
        "itemSelector": ".grid-item",
        "columnWidth": ".grid-sizer"
    });
    // layout Masonry after each image loads
    $grid.imagesLoaded().progress(function () {
        $grid.masonry('layout');
    });

    $(".lazy").on("load", function () {
        $grid.masonry('layout');
    });

    // Hover large image preview for publication covers (desktop only)
    var $hoverPreview = $('#image-hover-preview');
    if ($hoverPreview.length === 0) {
        $hoverPreview = $('<div id="image-hover-preview" aria-hidden="true"><img alt="preview"></div>');
        $('body').append($hoverPreview);
    }

    var previewOffsetX = 16;
    var previewOffsetY = 16;

    $(document).on('mouseenter', 'img.publication-cover', function (e) {
        var full = $(this).attr('data-full') || $(this).attr('data-src') || $(this).attr('src');
        if (!full) return;
        $hoverPreview.find('img').attr('src', full);
        $hoverPreview.css({ display: 'block' });
    });

    $(document).on('mousemove', 'img.publication-cover', function (e) {
        var vpW = $(window).width();
        var vpH = $(window).height();
        var previewW = $hoverPreview.outerWidth();
        var previewH = $hoverPreview.outerHeight();

        var x = e.clientX + previewOffsetX;
        var y = e.clientY + previewOffsetY;

        if (x + previewW > vpW - 8) x = e.clientX - previewW - previewOffsetX;
        if (y + previewH > vpH - 8) y = e.clientY - previewH - previewOffsetY;

        $hoverPreview.css({ left: x + 'px', top: y + 'px' });
    });

    $(document).on('mouseleave', 'img.publication-cover', function () {
        $hoverPreview.css({ display: 'none' });
        $hoverPreview.find('img').attr('src', '');
    });
})
