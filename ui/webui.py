import gradio as gr

from core.coreutils import image_upload, process_image_click

from core import TaskEraseScratches, TaskSuperFace, TaskImageColorizer, \
                TaskLowLight, TaskZeroBackground, \
                TaskTextToImage, TaskGenerativeFill, \
                TaskReplaceBackground, TaskRemoveSelection

class InfiniteImagination:
    def __init__(self):

        self.erase_scratches = TaskEraseScratches()
        self.enhance_face = TaskSuperFace()
        self.colorize_photo = TaskImageColorizer()
        self.enhance_light = TaskLowLight()
        self.remove_background = TaskZeroBackground()

        self.text_to_image = TaskTextToImage()
        self.generative_fill = TaskGenerativeFill()
        self.replace_background = TaskReplaceBackground()
        self.remove_selection = TaskRemoveSelection()
    
    ################### IMAGE ENHANCER METHODS ###################
    def eraseScratches(self, image):
        new_img = self.erase_scratches.executeTask(image)
        new_img = self.enhance_face.executeTask(new_img)
        return new_img

    def enhanceFace(self, image):
        new_img = self.enhance_face.executeTask(image)
        return new_img

    def colorizePhoto(self, image):
        new_img = self.colorize_photo.executeTask(image)
        return new_img

    def enhanceLight(self, image):
        new_img = self.enhance_light.executeTask(image)
        new_img = self.enhance_face.executeTask(new_img)
        return new_img

    def removeBackground(self, image):
        new_img = self.remove_background.executeTask(image)
        return new_img

    ################### GENERATIVE AI METHODS ###################
    def generateImage(self, payload, image_resolution, clicked_points):
        image = self.text_to_image.executeTask(payload)
        new_img, features, orig_h, orig_w, input_h, input_w, clicked_points = self.uploadImage(image, image_resolution, clicked_points)
        return new_img, new_img, features, orig_h, orig_w, input_h, input_w, clicked_points, None, None

    def generativeFill(self, origin_image, click_mask, image_resolution, adv_text_prompt):
        new_img = self.generative_fill.executeTask(origin_image, click_mask, image_resolution, adv_text_prompt)
        return new_img

    def replaceBackground(self, origin_image, click_mask, image_resolution, adv_text_prompt):
        new_img = self.replace_background.executeTask(origin_image, click_mask, image_resolution, adv_text_prompt)
        return new_img

    def removeSelection(self, origin_image, click_mask, image_resolution):
        new_img = self.remove_selection.executeTask(origin_image, click_mask, image_resolution)
        return new_img

    ################### IMAGE UPLOAD AND SELECTION METHODS ###################
    def uploadImage(self, image, image_resolution, clicked_points):
        return image_upload(image, image_resolution, clicked_points)

    def selectImage(self, original_image, point_prompt, clicked_points, image_resolution, features, orig_h, orig_w, input_h, input_w, dilate_kernel_size, evt: gr.SelectData):
        return process_image_click(original_image, point_prompt, clicked_points, image_resolution, features, orig_h, orig_w, input_h, input_w, dilate_kernel_size, evt.index)
    
    ################### GENERAL METHODS ###################
    def mirrorImage(self, input_image, output_image):
        return output_image, input_image

    def swapImage(self, input_image, output_image, clicked_points):
        clicked_points.clear()
        return output_image, input_image, None, None, None, None, clicked_points
    
    def clearImages(self):
        return None, None

    def resetImages(self, clicked_points, *args):
        clicked_points.clear()
        reset_arr = [None for _ in args]
        reset_arr.append(clicked_points)
        # print("Reset Array is: ", reset_arr)
        return reset_arr

    ################### GRADIO WEB UI ###################
    def launch(self):
        css = """
        .controlbtn-cls {background: #73D393}
        #controlbtn-id {background: #73D393}
        .clearbtn-cls {background: #FF7F50; color: white}
        #clearbtn-id {background: #FF7F50; color: white}
        .input-image-cls {height: 400px !important}
        .output-image-cls {height: 400px !important}
        #controlbtn-margin-id {margin-top: 40px;background: #73D393}
        """
        # Setup Gradio interface for refined images
        # title="Infinite Imagination"
        with gr.Blocks(theme=gr.themes.Soft(), css=css) as demo:
            logger = gr.State(value=[])
            clicked_points = gr.State([])
            origin_image = gr.State(None)
            click_mask = gr.State(None)
            features = gr.State(None)
            orig_h = gr.State(None)
            orig_w = gr.State(None)
            input_h = gr.State(None)
            input_w = gr.State(None)

            with gr.Row():
                with gr.Column(scale=2):
                    with gr.Row():
                        #gr.Markdown("# ![](file=images/logo.png)")
                        gr.HTML("<div style='display: flex; align-items: center; justify-content: center;'><div style=''><img style='max-width: 100%; max-height:100%;' src='file=images/logo.png'></div><div style='padding-left: 14px;'><h1>Infinite Imagination</h1></div></div>")

            with gr.Tabs() as tabs:
                with gr.TabItem("Image Enhancer", id=0):
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=1):
                            btn_eraser = gr.Button(value="Erase Scratches", elem_id="controlbtn-margin-id", elem_classes="controlbtn-cls")
                            btn_hires = gr.Button(value="Enhance Face", elem_id="controlbtn-id", elem_classes="controlbtn-cls")
                            btn_color = gr.Button(value="Colorize Photo", elem_id="controlbtn-id", elem_classes="controlbtn-cls")
                            btn_light = gr.Button(value="Enhance Light", elem_id="controlbtn-id", elem_classes="controlbtn-cls")
                            btn_clear = gr.Button(value="Remove Background", elem_id="controlbtn-id", elem_classes="controlbtn-cls")
                        with gr.Column(scale=2):
                            with gr.Row(equal_height=False):
                                img_input = gr.Image(label="Input Image", type="numpy", elem_classes="input-image-cls")
                            with gr.Row(equal_height=False):
                                btn_reset = gr.Button(value="Clear", elem_id="clearbtn-id", elem_classes="clearbtn-cls")
                                btn_swap = gr.Button(value="<< Swap >>", variant="primary")
                        with gr.Column(scale=2):
                            with gr.Row(equal_height=False):
                                img_output = gr.Image(label="Changed Image", type="numpy", interactive=False, elem_classes="input-image-cls")
                with gr.TabItem("Generative AI", id=1):
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=1):
                            dilate_kernel_size = gr.Slider(label="Dilate Kernel Size", minimum=0, maximum=100, step=1, value=0, visible=False)
                            point_prompt = gr.Radio(
                                choices=["Foreground Point","Background Point"], value="Foreground Point", label="Point Label",
                                interactive=True, show_label=False, visible=False
                            )
                            image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=768, step=64, value=512, visible=False)
                            adv_text_prompt = gr.Textbox(label="Input Text", lines=2)
                            adv_btn_gen = gr.Button(value="Text to Image", elem_id="controlbtn-id", elem_classes="controlbtn-cls")
                            adv_btn_gen_fill_image = gr.Button("Generative Fill", elem_id="controlbtn-id", elem_classes="controlbtn-cls")
                            adv_btn_replace_bg_image = gr.Button("Replace Background", elem_id="controlbtn-id", elem_classes="controlbtn-cls")
                            adv_btn_remove_sel_image = gr.Button("Remove Selection", elem_id="controlbtn-id", elem_classes="controlbtn-cls")

                        with gr.Column(scale=2):
                            with gr.Row(equal_height=False):
                                adv_img = gr.Image(label="Input Image", type="numpy", elem_classes="input-image-cls")
                            with gr.Row(equal_height=False):
                                adv_btn_reset_image = gr.Button("Clear", elem_id="clearbtn-id", elem_classes="clearbtn-cls")
                                adv_btn_swap = gr.Button(value="<< Swap >>", variant="primary")
                        with gr.Column(scale=2):
                            with gr.Row(equal_height=False):
                                adv_img_rm_with_mask = gr.Image(type="numpy", label="Changed Image", interactive=False, elem_classes="output-image-cls")


        btn_eraser.click(self.eraseScratches, inputs=[img_input], outputs=[img_output])
        btn_hires.click(self.enhanceFace, inputs=[img_input], outputs=[img_output])
        btn_color.click(self.colorizePhoto, inputs=[img_input], outputs=[img_output])
        btn_light.click(self.enhanceLight, inputs=[img_input], outputs=[img_output])
        btn_clear.click(self.removeBackground, inputs=[img_input], outputs=[img_output])

        btn_reset.click(self.clearImages, outputs=[img_input, img_output])
        btn_swap.click(self.mirrorImage, inputs=[img_input,img_output], outputs=[img_input,img_output])

        adv_img.upload(
            self.uploadImage,
            inputs=[adv_img, image_resolution, clicked_points],
            outputs=[origin_image, features, orig_h, orig_w, input_h, input_w, clicked_points],
        )

        adv_img.select(
            self.selectImage,
            inputs=[origin_image, point_prompt, clicked_points, image_resolution, features, orig_h, orig_w, input_h, input_w, dilate_kernel_size],
            outputs=[adv_img, clicked_points, click_mask],
            show_progress=True,
            queue=True,
        )

        adv_btn_gen.click(
            self.generateImage,
            inputs=[adv_text_prompt, image_resolution, clicked_points],
            outputs=[adv_img, origin_image, features, orig_h, orig_w, input_h, input_w, clicked_points, adv_img_rm_with_mask, click_mask]
        )

        adv_btn_gen_fill_image.click(
            self.generativeFill,
            inputs=[origin_image, click_mask, image_resolution, adv_text_prompt],
            outputs=[adv_img_rm_with_mask]
        )

        adv_btn_replace_bg_image.click(
            self.replaceBackground,
            inputs=[origin_image, click_mask, image_resolution, adv_text_prompt],
            outputs=[adv_img_rm_with_mask]
        )

        adv_btn_remove_sel_image.click(
            self.removeSelection,
            inputs=[origin_image, click_mask, image_resolution],
            outputs=[adv_img_rm_with_mask]
        )

        adv_btn_reset_image.click(
            self.resetImages,
            inputs=[clicked_points, adv_img, features, adv_img_rm_with_mask, adv_text_prompt, origin_image, click_mask],
            outputs=[adv_img, features, adv_img_rm_with_mask, adv_text_prompt, origin_image, click_mask, clicked_points]
        )

        adv_btn_swap.click(
            self.swapImage,
            inputs=[origin_image, adv_img_rm_with_mask, clicked_points],
            outputs=[adv_img, adv_img_rm_with_mask, features, adv_text_prompt, origin_image, click_mask, clicked_points]
        )
        # Launch the app
        demo.queue().launch(show_api=False, share=True, debug=False)
