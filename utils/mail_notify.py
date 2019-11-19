import os
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header
from email.utils import formataddr
from utils.trivial_definition import separator_line
import argparse
from utils.log_recorder import verbose_logs
from configs.param_config import ConfigClass
from options.args_option import set_argparse_opts


mail_templete = """
    <p>{summary_lines:s}</p>
    <p>{last_line_in_logs:s}<p>
    <p>{time_consume_line:s}</p>
    <br>
    <p>Online Links of Tensorboard is:</p>
    {links_html:s}
    <br>
    <p>Please refer to the attached documents for details</p>
    <p>------------------------------</p>
"""

def send_mail_notification(args, mail_config_file=None):
    if mail_config_file is None:
        mail_config_file = os.path.dirname(os.path.realpath(__file__)) + "/mail_config.json"

    with open(mail_config_file, "r") as mcf:
        mail_param = json.load(mcf)

        attachments_list = [attachment.format(log_name=args.log_name) for attachment in mail_param["attachments"]]

        mail_subject = "[{code:s}] Result of {net_arch:s} on {data_name:s}".format(code=args.code,
                                                                                   net_arch=args.net_arch,
                                                                                   data_name=args.dataset)
        summary_lines = "Summary of running <b>{code:s}</b> "\
                        "with {net_arch:s} on {data_name:s}:".format(code=args.code,
                                                                        net_arch=args.net_arch,
                                                                        data_name=args.dataset)

        time_consume_line = "Time elapsed {:.2f} hours.".format(args.running_time)

        from_nickname = mail_param["from_nickname"].format(code=args.code)

        # read the last line of verbose logs as mail content text
        with open(attachments_list[0], 'r') as flog:
            lines = flog.readlines()
            last_line_in_logs = lines[-1]

        # tensor_board links
        links_html = ""
        for (text_str, herf_str) in mail_param["online_links"].items():
            herf_str = herf_str.format(log_name=args.log_name.split("/")[-1])
            link_html = """
                <p><a href="{herf_str:s}">{text_str:s}</a></p>
            """.format(herf_str=herf_str, text_str=text_str)
            links_html += link_html

        mail_content = mail_templete.format(summary_lines=summary_lines,
                                            last_line_in_logs=last_line_in_logs,
                                            time_consume_line=time_consume_line,
                                            links_html=links_html)

        try:
            # create a e-mail header
            message = MIMEMultipart()
            message['From'] = formataddr((Header(from_nickname, "utf-8").encode(),
                                          mail_param["mail_username"]))

            message['To'] = ",".join(mail_param["target_addresses"])

            message['Subject'] = Header(mail_subject, "utf-8")

            # mail content html
            message.attach(MIMEText(mail_content, "html", "utf-8"))

            valid_attachments = []
            # mail attachments
            for att_name in attachments_list:
                file_name = att_name.split("/")[-1]
                if not os.path.isfile(att_name):
                    stream_str = "Fail to read '{:s}'".format(file_name)
                    message.attach(MIMEText(stream_str + "\r\n", "plain", "utf-8"))
                    print("=>" + stream_str)
                else:
                    valid_attachments.append(att_name)

            # message.attach(MIMEText("-" * 40 + "\r\n", "plain", "utf-8"))
            if len(valid_attachments) != len(attachments_list):
                print(separator_line())

            for att_name in valid_attachments:
                file_name = att_name.split("/")[-1]
                attachment = MIMEText(open(att_name, "rb").read(), "base64", "utf-8")
                attachment["Content-Type"] = "application/octet-stream"
                attachment["Content-Disposition"] = "attachment; filename={:s}".format(file_name)
                message.attach(attachment)

            # smtp handle
            server = smtplib.SMTP_SSL(mail_param["ssl_server"], mail_param["ssl_port"])
            server.login(mail_param["mail_username"], mail_param["mail_password"])
            server.sendmail(mail_param["mail_username"],
                            mail_param["target_addresses"],
                            message.as_string())

            print("Email notification has been send successfully.")

        except Exception:
            print("Error: Fail to send Email.")

        print(separator_line())


if __name__ == "__main__":

    mail_config_file = os.path.dirname(os.path.realpath(__file__)) + "/mail_config.json"

    with open(mail_config_file, "r") as mcf:
        mail_param = json.load(mcf)

    server = smtplib.SMTP_SSL(mail_param["ssl_server"], mail_param["ssl_port"])

    #
    # config = ConfigClass()
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--option', default=None, metavar='PATH',
    #                     type=str, help="args option file path (default: None)")
    #
    # parser.add_argument('--resume', default=None, metavar='PATH',
    #                     type=str, help="resume file path (default: None)")
    #
    # parser.add_argument('--code', default=None, type=str,
    #                     help="identify code for another checkpoint")
    #
    # parser.add_argument('--evaluate', default=None, action='store_true',
    #                     help="evaluate model from checkpoint")
    #
    # parser.add_argument('--pretrained', default=None, type=str,
    #                     help="identify code for another checkpoint")
    # args = parser.parse_args()
    # args, stream_pool_sub = set_argparse_opts(args)
    # path_param = config.get_path_param(args.dataset)
    # # create verbose logger
    # logger, log_name = verbose_logs(path_param["log_root"], logger_name=args.code)
    # args.log_name = log_name
    # args.running_time = 1024.0
    # send_mail_notification(args)
    pass
